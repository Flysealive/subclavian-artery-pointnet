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

def create_improved_dataset_loader(csv_file, numpy_dir, npoints=1024, batch_size=8):
    """
    Create dataset loader with improved labels and data quality
    """
    # Load improved labels
    df = pd.read_csv(csv_file)
    
    print(f"Loading dataset: {csv_file}")
    print(f"Dataset size: {len(df)}")
    print(f"Label distribution:")
    
    # Choose the best label type (cluster_label for meaningful classification)
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
    print(f"  Class 0: {label_dist.get(0, 0)} samples")
    print(f"  Class 1: {label_dist.get(1, 0)} samples")
    
    # Create temporary CSV with selected labels and fix filename extensions
    temp_csv = 'temp_improved_labels.csv'
    df_temp = df[['filename', label_col]].copy()
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
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, val_loader, test_loader

def train_improved_model(train_loader, val_loader, test_loader, epochs=25, model_name="improved"):
    """
    Train PointNet with improved data and better training loop
    """
    device = torch.device('cpu')
    
    # Model
    classifier = PointNetCls(k=2, feature_transform=False)
    classifier = classifier.to(device)
    
    # Optimizer with better settings
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Training loop with validation
    best_val_acc = 0
    train_losses = []
    val_accuracies = []
    
    print(f"\n=== TRAINING IMPROVED MODEL ===")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Model: {model_name}")
    
    for epoch in range(epochs):
        # Training phase
        classifier.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx, (points, target) in enumerate(train_loader):
            target = target[:, 0]  # Remove extra dimension
            points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)
            
            optimizer.zero_grad()
            pred, trans, trans_feat = classifier(points)
            loss = F.nll_loss(pred, target)
            loss.backward()
            optimizer.step()
            
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum().item()
            
            epoch_loss += loss.item()
            epoch_correct += correct
            epoch_total += points.size(0)
        
        train_acc = epoch_correct / epoch_total
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation phase
        classifier.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for points, target in val_loader:
                target = target[:, 0]
                points = points.transpose(2, 1)
                points, target = points.to(device), target.to(device)
                
                pred, _, _ = classifier(points)
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum().item()
                
                val_correct += correct
                val_total += points.size(0)
                val_loss += loss.item()
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), f'cls/{model_name}_best_model.pth')
        
        # Save periodic checkpoints
        if epoch % 5 == 0:
            torch.save(classifier.state_dict(), f'cls/{model_name}_epoch_{epoch}.pth')
        
        print(f'Epoch {epoch:2d}: Loss={avg_loss:.4f}, Train_Acc={train_acc:.4f}, Val_Acc={val_acc:.4f}, LR={optimizer.param_groups[0]["lr"]:.6f}')
    
    # Final test evaluation
    print(f"\n=== FINAL TEST EVALUATION ===")
    
    # Load best model
    classifier.load_state_dict(torch.load(f'cls/{model_name}_best_model.pth', map_location='cpu'))
    classifier.eval()
    
    test_correct = 0
    test_total = 0
    class_correct = [0, 0]
    class_total = [0, 0]
    
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
                
                class_total[label] += 1
                test_total += 1
                
                if prediction == label:
                    class_correct[label] += 1
                    test_correct += 1
    
    test_acc = test_correct / test_total if test_total > 0 else 0
    
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}")
    
    for i in range(2):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            print(f"Class {i} accuracy: {class_acc:.4f} ({class_correct[i]}/{class_total[i]})")
    
    return {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='high_quality', choices=['clean', 'balanced', 'high_quality'])
    parser.add_argument('--epochs', type=int, default=25)
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
    
    print(f"=== IMPROVED POINTNET TRAINING ===")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Points per cloud: {args.num_points}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_improved_dataset_loader(
        csv_file=csv_file,
        numpy_dir='numpy_arrays',
        npoints=args.num_points,
        batch_size=args.batch_size
    )
    
    # Train model
    results = train_improved_model(
        train_loader, val_loader, test_loader, 
        epochs=args.epochs, 
        model_name=f"{args.dataset}_improved"
    )
    
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"✅ Best validation accuracy: {results['best_val_acc']:.4f}")
    print(f"✅ Final test accuracy: {results['test_acc']:.4f}")
    print(f"✅ Models saved in cls/ directory")