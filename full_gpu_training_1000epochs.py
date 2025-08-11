#!/usr/bin/env python3
"""
Full GPU Training with 1000 Epochs
Binary classification with 3 anatomical measurements
Optimized for NVIDIA 4060Ti 8GB
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from pathlib import Path
import time
from datetime import datetime
import json

class FullDatasetWithMeasurements(Dataset):
    def __init__(self, csv_file, numpy_dir='numpy_arrays', split='train', npoints=2048, augment=False):
        self.numpy_dir = numpy_dir
        self.npoints = npoints
        self.augment = augment and (split == 'train')
        
        # Load data
        df = pd.read_csv(csv_file)
        
        # Rename columns if needed
        if 'left_subclavian_diameter_mm' in df.columns:
            df = df.rename(columns={
                'left_subclavian_diameter_mm': 'left_subclavian',
                'aortic_arch_diameter_mm': 'aortic_arch',
                'angle_degrees': 'angle'
            })
        
        # Split data
        np.random.seed(42)
        n = len(df)
        indices = np.random.permutation(n)
        
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        
        if split == 'train':
            self.data = df.iloc[indices[:train_end]]
        elif split == 'val':
            self.data = df.iloc[indices[train_end:val_end]]
        else:
            self.data = df.iloc[indices[val_end:]]
        
        self.data = self.data.reset_index(drop=True)
        
        # Calculate class weights for balanced training
        labels = self.data['label'].values
        class_counts = np.bincount(labels)
        self.class_weights = len(labels) / (len(np.unique(labels)) * class_counts)
        
        # Normalize measurements
        self.mean_meas = np.array([
            df['left_subclavian'].mean(),
            df['aortic_arch'].mean(),
            df['angle'].mean()
        ], dtype=np.float32)
        
        self.std_meas = np.array([
            df['left_subclavian'].std() + 1e-6,
            df['aortic_arch'].std() + 1e-6,
            df['angle'].std() + 1e-6
        ], dtype=np.float32)
        
        print(f"{split.upper()}: {len(self.data)} samples")
        print(f"  Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        print(f"  Class weights: {self.class_weights}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load point cloud
        file_path = os.path.join(self.numpy_dir, row['filename'])
        points = np.load(file_path).astype(np.float32)
        
        # Sample points
        if points.shape[0] >= self.npoints:
            choice = np.random.choice(points.shape[0], self.npoints, replace=False)
        else:
            choice = np.random.choice(points.shape[0], self.npoints, replace=True)
        points = points[choice, :]
        
        # Normalize point cloud
        centroid = points.mean(axis=0)
        points -= centroid
        max_dist = np.sqrt((points ** 2).sum(axis=1).max())
        if max_dist > 0:
            points /= max_dist
        
        # Data augmentation
        if self.augment:
            # Random rotation around Y axis
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ], dtype=np.float32)
            points = points @ rotation_matrix
            
            # Random jitter
            points += np.random.normal(0, 0.01, size=points.shape).astype(np.float32)
            
            # Random scaling
            scale = np.random.uniform(0.9, 1.1)
            points *= scale
        
        # Normalize measurements
        measurements = np.array([
            row['left_subclavian'],
            row['aortic_arch'],
            row['angle']
        ], dtype=np.float32)
        measurements = (measurements - self.mean_meas) / self.std_meas
        
        return (
            torch.from_numpy(points),
            torch.from_numpy(measurements),
            int(row['label'])
        )

class FullPointNetWithMeasurements(nn.Module):
    """Full model with all parameters for GPU training"""
    def __init__(self, num_classes=2, num_measurements=3):
        super().__init__()
        
        # Spatial Transform Network
        self.stn_conv1 = nn.Conv1d(3, 64, 1)
        self.stn_conv2 = nn.Conv1d(64, 128, 1)
        self.stn_conv3 = nn.Conv1d(128, 1024, 1)
        self.stn_fc1 = nn.Linear(1024, 512)
        self.stn_fc2 = nn.Linear(512, 256)
        self.stn_fc3 = nn.Linear(256, 9)
        
        self.stn_bn1 = nn.BatchNorm1d(64)
        self.stn_bn2 = nn.BatchNorm1d(128)
        self.stn_bn3 = nn.BatchNorm1d(1024)
        self.stn_bn4 = nn.BatchNorm1d(512)
        self.stn_bn5 = nn.BatchNorm1d(256)
        
        # Point Feature Extraction
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)
        
        # Measurement Processing Network
        self.meas_fc1 = nn.Linear(num_measurements, 64)
        self.meas_fc2 = nn.Linear(64, 128)
        self.meas_fc3 = nn.Linear(128, 256)
        
        self.meas_bn1 = nn.BatchNorm1d(64)
        self.meas_bn2 = nn.BatchNorm1d(128)
        self.meas_bn3 = nn.BatchNorm1d(256)
        
        # Combined Classifier
        self.fc1 = nn.Linear(1024 + 256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.bn_fc3 = nn.BatchNorm1d(128)
        
        self.dropout = nn.Dropout(0.4)
        
        # Initialize STN to identity
        self.stn_fc3.weight.data.zero_()
        self.stn_fc3.bias.data = torch.tensor([1,0,0,0,1,0,0,0,1], dtype=torch.float32).view(-1)
        
    def stn(self, x):
        """Spatial Transformer Network"""
        batchsize = x.size()[0]
        x = F.relu(self.stn_bn1(self.stn_conv1(x)))
        x = F.relu(self.stn_bn2(self.stn_conv2(x)))
        x = F.relu(self.stn_bn3(self.stn_conv3(x)))
        x = torch.max(x, 2)[0]
        
        x = F.relu(self.stn_bn4(self.stn_fc1(x)))
        x = F.relu(self.stn_bn5(self.stn_fc2(x)))
        x = self.stn_fc3(x)
        
        iden = torch.eye(3, device=x.device).view(1, 9).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x
        
    def forward(self, points, measurements):
        # points: (B, N, 3)
        # measurements: (B, 3)
        
        B, N, _ = points.size()
        
        # Apply spatial transformer
        trans = self.stn(points.transpose(2, 1))
        points = points @ trans
        points = points.transpose(2, 1)  # (B, 3, N)
        
        # Extract point features
        x = F.relu(self.bn1(self.conv1(points)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Global max pooling
        x = torch.max(x, 2)[0]  # (B, 1024)
        
        # Process measurements
        m = F.relu(self.meas_bn1(self.meas_fc1(measurements)))
        m = F.relu(self.meas_bn2(self.meas_fc2(m)))
        m = F.relu(self.meas_bn3(self.meas_fc3(m)))
        
        # Combine features
        combined = torch.cat([x, m], dim=1)  # (B, 1024+256)
        
        # Classification
        x = F.relu(self.bn_fc1(self.fc1(combined)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        
        return F.log_softmax(x, dim=1), trans

def train_gpu_1000_epochs():
    print("="*70)
    print("FULL GPU TRAINING - 1000 EPOCHS")
    print("Binary Classification with 3 Anatomical Measurements")
    print("="*70)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        print(f"GPU DETECTED: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("WARNING: GPU not available, using CPU (will be slow)")
    
    print(f"Device: {device}")
    print()
    
    # Create datasets
    train_dataset = FullDatasetWithMeasurements(
        'classification_labels_with_measurements.csv', 
        split='train', 
        npoints=2048,
        augment=True
    )
    val_dataset = FullDatasetWithMeasurements(
        'classification_labels_with_measurements.csv', 
        split='val', 
        npoints=2048,
        augment=False
    )
    test_dataset = FullDatasetWithMeasurements(
        'classification_labels_with_measurements.csv', 
        split='test', 
        npoints=2048,
        augment=False
    )
    
    # Get class weights for balanced loss
    class_weights = torch.FloatTensor(train_dataset.class_weights).to(device)
    
    # Dataloaders
    batch_size = 32 if torch.cuda.is_available() else 8
    num_workers = 4 if torch.cuda.is_available() else 0
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available()
    )
    
    # Model
    model = FullPointNetWithMeasurements().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")
    print()
    
    # Loss and Optimizer
    criterion = nn.NLLLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    
    # Training variables
    best_val_acc = 0
    best_balanced_acc = 0
    best_epoch = 0
    patience = 100
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 
               'val_balanced_acc': [], 'lr': []}
    
    os.makedirs('cls', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    print("Starting 1000 Epoch Training...")
    print("-" * 70)
    
    start_time = time.time()
    
    for epoch in range(1000):
        epoch_start = time.time()
        
        # Training Phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (points, measurements, labels) in enumerate(train_loader):
            points = points.to(device)
            measurements = measurements.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs, trans = model(points, measurements)
            loss = criterion(outputs, labels)
            
            # Add regularization for transform
            reg_loss = 0.001 * torch.norm(torch.eye(3, device=device).unsqueeze(0) - trans @ trans.transpose(1, 2))
            loss = loss + reg_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if torch.cuda.is_available() and batch_idx % 10 == 0:
                print(f'  Epoch {epoch+1} Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}', end='\r')
        
        # Validation Phase
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
                labels = labels.to(device)
                
                outputs, _ = model(points, measurements)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    pred = predicted[i].item()
                    val_class_total[label] += 1
                    val_total += 1
                    if pred == label:
                        val_class_correct[label] += 1
                        val_correct += 1
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        val_acc = 100. * val_correct / max(val_total, 1)
        val_loss = val_loss / len(val_loader)
        
        val_c0_acc = 100. * val_class_correct[0] / max(val_class_total[0], 1)
        val_c1_acc = 100. * val_class_correct[1] / max(val_class_total[1], 1)
        val_balanced_acc = (val_c0_acc + val_c1_acc) / 2
        
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_balanced_acc'].append(val_balanced_acc)
        history['lr'].append(current_lr)
        
        # Print progress
        print(f'Epoch {epoch+1:4d}/1000: '
              f'Train[L={train_loss:.4f}, A={train_acc:.1f}%] '
              f'Val[L={val_loss:.4f}, A={val_acc:.1f}%, BA={val_balanced_acc:.1f}%] '
              f'C0={val_c0_acc:.1f}% C1={val_c1_acc:.1f}% '
              f'LR={current_lr:.6f} T={epoch_time:.1f}s')
        
        # Save best model
        if val_balanced_acc > best_balanced_acc:
            best_balanced_acc = val_balanced_acc
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_balanced_acc': val_balanced_acc,
            }, 'cls/gpu_1000epochs_best.pth')
            print(f'  --> New best model! Balanced Acc: {val_balanced_acc:.1f}%')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, f'cls/checkpoint_epoch_{epoch+1}.pth')
            print(f'  Checkpoint saved at epoch {epoch+1}')
        
        # Save history
        if (epoch + 1) % 10 == 0:
            with open('logs/training_history.json', 'w') as f:
                json.dump(history, f)
        
        scheduler.step()
        
        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch+1} (patience={patience})')
            break
    
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print(f"Training completed in {total_time/3600:.2f} hours")
    print(f"Best epoch: {best_epoch} with balanced accuracy: {best_balanced_acc:.1f}%")
    
    # Test Evaluation
    print("\n" + "="*70)
    print("FINAL TEST EVALUATION")
    print("="*70)
    
    checkpoint = torch.load('cls/gpu_1000epochs_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_correct = 0
    test_total = 0
    class_correct = [0, 0]
    class_total = [0, 0]
    confusion_matrix = torch.zeros(2, 2)
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for points, measurements, labels in test_loader:
            points = points.to(device)
            measurements = measurements.to(device)
            labels = labels.to(device)
            
            outputs, _ = model(points, measurements)
            probs = torch.exp(outputs)
            _, predicted = outputs.max(1)
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                prob = probs[i].cpu().numpy()
                
                all_labels.append(label)
                all_predictions.append(pred)
                all_probs.append(prob)
                
                confusion_matrix[label, pred] += 1
                class_total[label] += 1
                test_total += 1
                
                if pred == label:
                    class_correct[label] += 1
                    test_correct += 1
    
    # Calculate final metrics
    test_acc = 100. * test_correct / test_total
    test_balanced_acc = 50. * (class_correct[0]/max(class_total[0],1) + 
                               class_correct[1]/max(class_total[1],1))
    
    print(f"\nOverall Test Accuracy: {test_acc:.1f}% ({test_correct}/{test_total})")
    print(f"Balanced Test Accuracy: {test_balanced_acc:.1f}%")
    
    for i in range(2):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            print(f"Class {i}: {acc:.1f}% ({class_correct[i]}/{class_total[i]})")
    
    print("\nConfusion Matrix:")
    print("       Pred_0  Pred_1")
    print(f"True_0:  {int(confusion_matrix[0,0]):4d}   {int(confusion_matrix[0,1]):4d}")
    print(f"True_1:  {int(confusion_matrix[1,0]):4d}   {int(confusion_matrix[1,1]):4d}")
    
    # Save final results
    results = {
        'test_accuracy': test_acc,
        'test_balanced_accuracy': test_balanced_acc,
        'class_0_accuracy': 100. * class_correct[0] / max(class_total[0], 1),
        'class_1_accuracy': 100. * class_correct[1] / max(class_total[1], 1),
        'confusion_matrix': confusion_matrix.tolist(),
        'best_epoch': best_epoch,
        'total_epochs': epoch + 1,
        'training_time_hours': total_time / 3600,
        'device': str(device),
        'model_parameters': total_params
    }
    
    with open('logs/final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print(f"Results saved to logs/final_results.json")
    print(f"Best model saved to cls/gpu_1000epochs_best.pth")
    print("="*70)

if __name__ == "__main__":
    train_gpu_1000_epochs()