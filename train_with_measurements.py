#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
from subclavian_dataset_with_measurements import SubclavianDatasetWithMeasurements
from pointnet_with_measurements import PointNetWithMeasurements, PointNetWithMeasurementsEarlyFusion

def train_model_with_measurements(model_type='late_fusion', epochs=100, batch_size=8):
    """
    Train PointNet with anatomical measurements
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = SubclavianDatasetWithMeasurements(
        numpy_dir='numpy_arrays',
        csv_file='classification_labels_with_measurements.csv',
        split='train',
        npoints=1024,
        data_augmentation=True
    )
    
    val_dataset = SubclavianDatasetWithMeasurements(
        numpy_dir='numpy_arrays',
        csv_file='classification_labels_with_measurements.csv',
        split='val',
        npoints=1024,
        data_augmentation=False
    )
    
    test_dataset = SubclavianDatasetWithMeasurements(
        numpy_dir='numpy_arrays',
        csv_file='classification_labels_with_measurements.csv',
        split='test',
        npoints=1024,
        data_augmentation=False
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    if model_type == 'late_fusion':
        model = PointNetWithMeasurements(k=2, num_measurements=3)
    else:  # early_fusion
        model = PointNetWithMeasurementsEarlyFusion(k=2, num_measurements=3)
    
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    # Training loop
    best_val_acc = 0
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 30
    
    print(f"\n=== TRAINING {model_type.upper()} MODEL WITH MEASUREMENTS ===")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for points, measurements, labels in train_loader:
            points = points.to(device)
            measurements = measurements.to(device)
            labels = labels.squeeze().to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred, trans, trans_feat = model(points, measurements)
            loss = F.nll_loss(pred, labels)
            
            # Add regularization for feature transform
            if trans_feat is not None:
                loss += 0.001 * feature_transform_regularizer(trans_feat)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred_choice = pred.data.max(1)[1]
            train_correct += pred_choice.eq(labels.data).cpu().sum().item()
            train_total += labels.size(0)
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_class_correct = [0, 0]
        val_class_total = [0, 0]
        
        with torch.no_grad():
            for points, measurements, labels in val_loader:
                points = points.to(device)
                measurements = measurements.to(device)
                labels = labels.squeeze().to(device)
                
                pred, _, _ = model(points, measurements)
                loss = F.nll_loss(pred, labels)
                
                val_loss += loss.item()
                pred_choice = pred.data.max(1)[1]
                
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    prediction = pred_choice[i].item()
                    val_class_total[label] += 1
                    val_total += 1
                    
                    if prediction == label:
                        val_class_correct[label] += 1
                        val_correct += 1
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        # Calculate per-class accuracy
        val_class_acc = []
        for i in range(2):
            if val_class_total[i] > 0:
                acc = val_class_correct[i] / val_class_total[i]
                val_class_acc.append(acc)
            else:
                val_class_acc.append(0)
        
        balanced_acc = np.mean(val_class_acc)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if balanced_acc > best_val_acc:
            best_val_acc = balanced_acc
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'cls/{model_type}_with_measurements_best.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f'Epoch {epoch+1:3d}: Train[Loss={avg_train_loss:.4f}, Acc={train_acc:.3f}], '
              f'Val[Loss={avg_val_loss:.4f}, Acc={val_acc:.3f}, C0={val_class_acc[0]:.3f}, '
              f'C1={val_class_acc[1]:.3f}, Balanced={balanced_acc:.3f}]')
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Test evaluation
    print(f"\n=== FINAL TEST EVALUATION ===")
    
    # Load best model
    model.load_state_dict(torch.load(f'cls/{model_type}_with_measurements_best.pth', map_location=device))
    model.eval()
    
    test_correct = 0
    test_total = 0
    class_correct = [0, 0]
    class_total = [0, 0]
    confusion = np.zeros((2, 2), dtype=int)
    
    with torch.no_grad():
        for points, measurements, labels in test_loader:
            points = points.to(device)
            measurements = measurements.to(device)
            labels = labels.squeeze().to(device)
            
            pred, _, _ = model(points, measurements)
            pred_choice = pred.data.max(1)[1]
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                prediction = pred_choice[i].item()
                
                confusion[label, prediction] += 1
                class_total[label] += 1
                test_total += 1
                
                if prediction == label:
                    class_correct[label] += 1
                    test_correct += 1
    
    test_acc = test_correct / test_total if test_total > 0 else 0
    
    print(f"Best validation balanced accuracy: {best_val_acc:.4f}")
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
    
    return {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'balanced_test_acc': balanced_test_acc,
        'confusion_matrix': confusion
    }


def compare_models():
    """
    Compare performance with and without measurements
    """
    print("="*60)
    print("COMPARING MODELS WITH AND WITHOUT MEASUREMENTS")
    print("="*60)
    
    # Train without measurements (baseline)
    print("\n1. BASELINE: PointNet without measurements")
    print("-"*40)
    # You would run your original training here
    # baseline_results = train_original_pointnet()
    
    # Train with measurements (late fusion)
    print("\n2. ENHANCED: PointNet with measurements (Late Fusion)")
    print("-"*40)
    late_fusion_results = train_model_with_measurements(model_type='late_fusion', epochs=100)
    
    # Train with measurements (early fusion)
    print("\n3. ENHANCED: PointNet with measurements (Early Fusion)")
    print("-"*40)
    early_fusion_results = train_model_with_measurements(model_type='early_fusion', epochs=100)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    print(f"Late Fusion - Test Accuracy: {late_fusion_results['test_acc']:.4f}, "
          f"Balanced: {late_fusion_results['balanced_test_acc']:.4f}")
    print(f"Early Fusion - Test Accuracy: {early_fusion_results['test_acc']:.4f}, "
          f"Balanced: {early_fusion_results['balanced_test_acc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='late_fusion', 
                       choices=['late_fusion', 'early_fusion'],
                       help='Type of fusion for measurements')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--compare', action='store_true',
                       help='Compare both fusion methods')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models()
    else:
        results = train_model_with_measurements(
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        print(f"\n✅ Training complete!")
        print(f"Final balanced test accuracy: {results['balanced_test_acc']:.4f}")