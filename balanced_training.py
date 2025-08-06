#!/usr/bin/env python3

import torch
import pandas as pd
import numpy as np
from subclavian_dataset import SubclavianDataset
import sys
sys.path.append('./pointnet.pytorch')
from pointnet.model import PointNetCls
import torch.nn.functional as F
import argparse
from torch.utils.data import WeightedRandomSampler
from sklearn.utils import resample
import torch.nn as nn

def create_balanced_dataset_loader(csv_file, numpy_dir, npoints=1024, batch_size=8, balance_method='weighted'):
    """
    Create dataset loader with class balancing
    balance_method: 'weighted', 'oversample', 'undersample', or 'none'
    """
    df = pd.read_csv(csv_file)
    
    print(f"Loading dataset: {csv_file}")
    print(f"Dataset size: {len(df)}")
    print(f"Balance method: {balance_method}")
    
    # Choose the best label type
    if 'cluster_label' in df.columns:
        label_col = 'cluster_label'
        print(f"  Using cluster-based labels")
    elif 'volume_label' in df.columns:
        label_col = 'volume_label'
        print(f"  Using volume-based labels")
    else:
        label_col = 'size_label'
        print(f"  Using size-based labels")
    
    label_dist = df[label_col].value_counts()
    print(f"Original distribution:")
    print(f"  Class 0: {label_dist.get(0, 0)} samples")
    print(f"  Class 1: {label_dist.get(1, 0)} samples")
    
    # Apply balancing strategy to training data
    if balance_method == 'oversample':
        # Oversample minority class
        df_majority = df[df[label_col] == 0]
        df_minority = df[df[label_col] == 1]
        
        # Oversample minority class
        df_minority_upsampled = resample(df_minority, 
                                        replace=True,
                                        n_samples=len(df_majority),
                                        random_state=42)
        
        df_balanced = pd.concat([df_majority, df_minority_upsampled])
        print(f"After oversampling:")
        print(f"  Class 0: {len(df_balanced[df_balanced[label_col] == 0])} samples")
        print(f"  Class 1: {len(df_balanced[df_balanced[label_col] == 1])} samples")
        
    elif balance_method == 'undersample':
        # Undersample majority class
        df_majority = df[df[label_col] == 0]
        df_minority = df[df[label_col] == 1]
        
        # Undersample majority class
        df_majority_downsampled = resample(df_majority, 
                                          replace=False,
                                          n_samples=len(df_minority),
                                          random_state=42)
        
        df_balanced = pd.concat([df_majority_downsampled, df_minority])
        print(f"After undersampling:")
        print(f"  Class 0: {len(df_balanced[df_balanced[label_col] == 0])} samples")
        print(f"  Class 1: {len(df_balanced[df_balanced[label_col] == 1])} samples")
    else:
        df_balanced = df
    
    # Create temporary CSV
    temp_csv = 'temp_balanced_labels.csv'
    df_temp = df_balanced[['filename', label_col]].copy()
    df_temp['filename'] = df_temp['filename'].str.replace('.stl', '.npy')
    df_temp.rename(columns={label_col: 'label'}).to_csv(temp_csv, index=False)
    
    # Create datasets
    train_dataset = SubclavianDataset(
        numpy_dir=numpy_dir,
        csv_file=temp_csv,
        split='train',
        npoints=npoints,
        train_ratio=0.7,
        val_ratio=0.15,
        data_augmentation=True
    )
    
    val_dataset = SubclavianDataset(
        numpy_dir=numpy_dir,
        csv_file=temp_csv,
        split='val',
        npoints=npoints,
        train_ratio=0.7,
        val_ratio=0.15,
        data_augmentation=False
    )
    
    test_dataset = SubclavianDataset(
        numpy_dir=numpy_dir,
        csv_file=temp_csv,
        split='test',
        npoints=npoints,
        train_ratio=0.7,
        val_ratio=0.15,
        data_augmentation=False
    )
    
    # Create weighted sampler for training if using weighted method
    if balance_method == 'weighted':
        # Calculate class weights for training set
        train_labels = []
        for i in range(len(train_dataset)):
            _, label = train_dataset[i]
            train_labels.append(label[0])
        
        train_labels = np.array(train_labels)
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        weights = class_weights[train_labels]
        
        sampler = WeightedRandomSampler(weights, len(weights))
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0
        )
        
        print(f"Using weighted sampling with class weights: {class_weights}")
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # Calculate class weights for loss function
    all_labels = df_balanced[label_col].values
    class_counts = np.bincount(all_labels)
    class_weights = len(all_labels) / (len(np.unique(all_labels)) * class_counts)
    class_weights = torch.FloatTensor(class_weights)
    
    return train_loader, val_loader, test_loader, class_weights

def focal_loss(pred, target, gamma=2.0, alpha=None):
    """
    Focal loss for addressing class imbalance
    """
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = (1 - pt) ** gamma * ce_loss
    
    if alpha is not None:
        alpha_t = alpha[target]
        focal_loss = alpha_t * focal_loss
    
    return focal_loss.mean()

def train_balanced_model(train_loader, val_loader, test_loader, class_weights=None, 
                        epochs=30, model_name="balanced", loss_type='weighted'):
    """
    Train PointNet with class balancing
    loss_type: 'weighted', 'focal', or 'standard'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    classifier = PointNetCls(k=2, feature_transform=False)
    classifier = classifier.to(device)
    
    # Move class weights to device
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    # Optimizer with better settings
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Training loop
    best_val_acc = 0
    best_val_f1 = 0
    train_losses = []
    val_accuracies = []
    
    print(f"\n=== TRAINING BALANCED MODEL ===")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Loss type: {loss_type}")
    print(f"Model: {model_name}")
    
    for epoch in range(epochs):
        # Training phase
        classifier.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        class_correct = [0, 0]
        class_total = [0, 0]
        
        for batch_idx, (points, target) in enumerate(train_loader):
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)
            
            optimizer.zero_grad()
            pred, trans, trans_feat = classifier(points)
            
            # Choose loss function
            if loss_type == 'weighted' and class_weights is not None:
                loss = F.cross_entropy(pred, target, weight=class_weights)
            elif loss_type == 'focal':
                loss = focal_loss(pred, target, alpha=class_weights)
            else:
                loss = F.nll_loss(pred, target)
            
            loss.backward()
            optimizer.step()
            
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum().item()
            
            # Track per-class accuracy
            for i in range(points.size(0)):
                label = target[i].item()
                prediction = pred_choice[i].item()
                class_total[label] += 1
                if prediction == label:
                    class_correct[label] += 1
            
            epoch_loss += loss.item()
            epoch_correct += correct
            epoch_total += points.size(0)
        
        train_acc = epoch_correct / epoch_total
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Calculate per-class training accuracy
        train_class_acc = []
        for i in range(2):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i]
                train_class_acc.append(acc)
            else:
                train_class_acc.append(0)
        
        # Validation phase
        classifier.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        val_class_correct = [0, 0]
        val_class_total = [0, 0]
        
        with torch.no_grad():
            for points, target in val_loader:
                target = target[:, 0]
                points = points.transpose(2, 1)
                points, target = points.to(device), target.to(device)
                
                pred, _, _ = classifier(points)
                
                if loss_type == 'weighted' and class_weights is not None:
                    loss = F.cross_entropy(pred, target, weight=class_weights)
                elif loss_type == 'focal':
                    loss = focal_loss(pred, target, alpha=class_weights)
                else:
                    loss = F.nll_loss(pred, target)
                
                pred_choice = pred.data.max(1)[1]
                
                for i in range(points.size(0)):
                    label = target[i].item()
                    prediction = pred_choice[i].item()
                    val_class_total[label] += 1
                    val_total += 1
                    
                    if prediction == label:
                        val_class_correct[label] += 1
                        val_correct += 1
                
                val_loss += loss.item()
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_accuracies.append(val_acc)
        
        # Calculate per-class validation accuracy and F1 score
        val_class_acc = []
        for i in range(2):
            if val_class_total[i] > 0:
                acc = val_class_correct[i] / val_class_total[i]
                val_class_acc.append(acc)
            else:
                val_class_acc.append(0)
        
        # Calculate balanced accuracy (average of per-class accuracies)
        balanced_acc = np.mean(val_class_acc)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model based on balanced accuracy
        if balanced_acc > best_val_f1:
            best_val_f1 = balanced_acc
            torch.save(classifier.state_dict(), f'cls/{model_name}_best_model.pth')
        
        # Save periodic checkpoints
        if epoch % 5 == 0:
            torch.save(classifier.state_dict(), f'cls/{model_name}_epoch_{epoch}.pth')
        
        print(f'Epoch {epoch:2d}: Loss={avg_loss:.4f}, Train=[Acc={train_acc:.3f}, C0={train_class_acc[0]:.3f}, C1={train_class_acc[1]:.3f}], '
              f'Val=[Acc={val_acc:.3f}, C0={val_class_acc[0]:.3f}, C1={val_class_acc[1]:.3f}, Balanced={balanced_acc:.3f}]')
    
    # Final test evaluation
    print(f"\n=== FINAL TEST EVALUATION ===")
    
    # Load best model
    classifier.load_state_dict(torch.load(f'cls/{model_name}_best_model.pth', map_location=device))
    classifier.eval()
    
    test_correct = 0
    test_total = 0
    class_correct = [0, 0]
    class_total = [0, 0]
    
    # Confusion matrix
    confusion = np.zeros((2, 2), dtype=int)
    
    with torch.no_grad():
        for points, target in test_loader:
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)
            
            pred, _, _ = classifier(points)
            pred_choice = pred.data.max(1)[1]
            
            for i in range(points.size(0)):
                label = target[i].item()
                prediction = pred_choice[i].item()
                
                confusion[label, prediction] += 1
                class_total[label] += 1
                test_total += 1
                
                if prediction == label:
                    class_correct[label] += 1
                    test_correct += 1
    
    test_acc = test_correct / test_total if test_total > 0 else 0
    
    print(f"Best validation balanced accuracy: {best_val_f1:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}")
    
    # Per-class metrics
    for i in range(2):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            print(f"Class {i} accuracy: {class_acc:.4f} ({class_correct[i]}/{class_total[i]})")
    
    # Balanced accuracy
    class_accs = [class_correct[i]/class_total[i] if class_total[i] > 0 else 0 for i in range(2)]
    balanced_test_acc = np.mean(class_accs)
    print(f"Balanced test accuracy: {balanced_test_acc:.4f}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print("       Pred_0  Pred_1")
    print(f"True_0   {confusion[0,0]:4d}   {confusion[0,1]:4d}")
    print(f"True_1   {confusion[1,0]:4d}   {confusion[1,1]:4d}")
    
    # Calculate precision, recall, F1
    if confusion[1,1] + confusion[0,1] > 0:
        precision_1 = confusion[1,1] / (confusion[1,1] + confusion[0,1])
    else:
        precision_1 = 0
    
    if confusion[1,1] + confusion[1,0] > 0:
        recall_1 = confusion[1,1] / (confusion[1,1] + confusion[1,0])
    else:
        recall_1 = 0
    
    if precision_1 + recall_1 > 0:
        f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1)
    else:
        f1_1 = 0
    
    print(f"\nClass 1 metrics:")
    print(f"  Precision: {precision_1:.4f}")
    print(f"  Recall: {recall_1:.4f}")
    print(f"  F1-Score: {f1_1:.4f}")
    
    return {
        'best_val_balanced_acc': best_val_f1,
        'test_acc': test_acc,
        'balanced_test_acc': balanced_test_acc,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'confusion_matrix': confusion
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='high_quality', choices=['clean', 'balanced', 'high_quality'])
    parser.add_argument('--balance_method', default='weighted', choices=['weighted', 'oversample', 'undersample', 'none'])
    parser.add_argument('--loss_type', default='weighted', choices=['weighted', 'focal', 'standard'])
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_points', type=int, default=1024)
    
    args = parser.parse_args()
    
    # Choose dataset
    dataset_files = {
        'clean': 'standardized_data/clean_improved_labels.csv',
        'balanced': 'standardized_data/balanced_improved_labels.csv', 
        'high_quality': 'standardized_data/high_quality_improved_labels.csv'
    }
    
    csv_file = dataset_files[args.dataset]
    
    print(f"=== BALANCED POINTNET TRAINING ===")
    print(f"Dataset: {args.dataset}")
    print(f"Balance method: {args.balance_method}")
    print(f"Loss type: {args.loss_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Points per cloud: {args.num_points}")
    
    # Create data loaders with balancing
    train_loader, val_loader, test_loader, class_weights = create_balanced_dataset_loader(
        csv_file=csv_file,
        numpy_dir='numpy_arrays',
        npoints=args.num_points,
        batch_size=args.batch_size,
        balance_method=args.balance_method
    )
    
    # Train model with balanced approach
    results = train_balanced_model(
        train_loader, val_loader, test_loader,
        class_weights=class_weights,
        epochs=args.epochs,
        model_name=f"{args.dataset}_{args.balance_method}_{args.loss_type}",
        loss_type=args.loss_type
    )
    
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"✅ Best validation balanced accuracy: {results['best_val_balanced_acc']:.4f}")
    print(f"✅ Final test accuracy: {results['test_acc']:.4f}")
    print(f"✅ Balanced test accuracy: {results['balanced_test_acc']:.4f}")
    print(f"✅ Models saved in cls/ directory")