#!/usr/bin/env python3
"""
CPU-optimized training script for PointNet with anatomical measurements
Designed for faster training on CPU with reduced model complexity
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

class FastDataset(Dataset):
    def __init__(self, csv_file, numpy_dir='numpy_arrays', split='train', npoints=512):
        """Reduced points for faster CPU training"""
        self.numpy_dir = numpy_dir
        self.npoints = npoints  # Reduced from 1024 to 512
        
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
        
        # Pre-compute normalization parameters
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
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load and process point cloud
        file_path = os.path.join(self.numpy_dir, row['filename'])
        points = np.load(file_path).astype(np.float32)
        
        # Fast point sampling
        n_pts = points.shape[0]
        if n_pts >= self.npoints:
            indices = np.random.choice(n_pts, self.npoints, replace=False)
        else:
            indices = np.random.choice(n_pts, self.npoints, replace=True)
        points = points[indices]
        
        # Fast normalization
        centroid = points.mean(axis=0)
        points -= centroid
        max_dist = np.sqrt((points ** 2).sum(axis=1).max())
        if max_dist > 0:
            points /= max_dist
        
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

class LightweightPointNet(nn.Module):
    """Simplified model for faster CPU training"""
    def __init__(self, num_classes=2, num_measurements=3):
        super().__init__()
        
        # Reduced channels for faster computation
        self.conv1 = nn.Conv1d(3, 32, 1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 256, 1)
        
        # Measurement processor
        self.meas_fc = nn.Linear(num_measurements, 16)
        
        # Classifier
        self.fc1 = nn.Linear(256 + 16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, points, measurements):
        # points: (B, N, 3)
        B, N, _ = points.size()
        x = points.transpose(2, 1)  # (B, 3, N)
        
        # Extract features
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Global pooling
        x = torch.max(x, 2)[0]  # (B, 256)
        
        # Process measurements
        m = F.relu(self.meas_fc(measurements))
        
        # Combine and classify
        combined = torch.cat([x, m], dim=1)
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)

def train_fast():
    print("=" * 60)
    print("CPU-OPTIMIZED TRAINING WITH MEASUREMENTS")
    print("=" * 60)
    
    device = torch.device('cpu')
    print(f"Device: {device}")
    print("Optimizations: Reduced points (512), Smaller model, Faster operations")
    
    # Create datasets with reduced points
    train_dataset = FastDataset('classification_labels_with_measurements.csv', split='train', npoints=512)
    val_dataset = FastDataset('classification_labels_with_measurements.csv', split='val', npoints=512)
    test_dataset = FastDataset('classification_labels_with_measurements.csv', split='test', npoints=512)
    
    # Optimized batch size for CPU
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create lightweight model
    model = LightweightPointNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} (lightweight)")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    
    # Training
    print("\nStarting fast training...")
    best_val_acc = 0
    best_model_state = None
    epochs = 30  # Reduced epochs
    
    train_start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for points, measurements, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(points, measurements)
            loss = F.nll_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_class_correct = [0, 0]
        val_class_total = [0, 0]
        
        with torch.no_grad():
            for points, measurements, labels in val_loader:
                outputs = model(points, measurements)
                loss = F.nll_loss(outputs, labels)
                
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
        val_acc = 100. * val_correct / max(val_total, 1)
        val_c0_acc = 100. * val_class_correct[0] / max(val_class_total[0], 1)
        val_c1_acc = 100. * val_class_correct[1] / max(val_class_total[1], 1)
        
        epoch_time = time.time() - epoch_start
        
        print(f'Epoch {epoch+1:2d}/{epochs}: '
              f'Train={train_acc:.1f}% '
              f'Val={val_acc:.1f}% (C0={val_c0_acc:.1f}%, C1={val_c1_acc:.1f}%) '
              f'Time={epoch_time:.1f}s')
        
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f'  --> New best model! Val accuracy: {val_acc:.1f}%')
    
    total_time = time.time() - train_start
    print(f"\nTraining completed in {total_time:.1f} seconds")
    
    # Save model
    if best_model_state is not None:
        os.makedirs('cls', exist_ok=True)
        torch.save(best_model_state, 'cls/cpu_optimized_model.pth')
        print("Model saved to cls/cpu_optimized_model.pth")
    
    # Test evaluation
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    model.load_state_dict(best_model_state)
    model.eval()
    
    test_correct = 0
    test_total = 0
    class_correct = [0, 0]
    class_total = [0, 0]
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for points, measurements, labels in test_loader:
            outputs = model(points, measurements)
            _, predicted = outputs.max(1)
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                
                all_labels.append(label)
                all_predictions.append(pred)
                
                class_total[label] += 1
                test_total += 1
                
                if pred == label:
                    class_correct[label] += 1
                    test_correct += 1
    
    # Print results
    overall_acc = 100. * test_correct / test_total
    print(f"\nOverall Test Accuracy: {overall_acc:.1f}% ({test_correct}/{test_total})")
    
    for i in range(2):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            print(f"Class {i} Accuracy: {acc:.1f}% ({class_correct[i]}/{class_total[i]})")
    
    # Confusion matrix
    from collections import Counter
    confusion = np.zeros((2, 2), dtype=int)
    for true, pred in zip(all_labels, all_predictions):
        confusion[true, pred] += 1
    
    print("\nConfusion Matrix:")
    print("       Pred_0  Pred_1")
    print(f"True_0:  {confusion[0,0]:3d}    {confusion[0,1]:3d}")
    print(f"True_1:  {confusion[1,0]:3d}    {confusion[1,1]:3d}")
    
    # Calculate balanced accuracy
    balanced_acc = 0.5 * (class_correct[0]/max(class_total[0], 1) + 
                          class_correct[1]/max(class_total[1], 1))
    print(f"\nBalanced Accuracy: {balanced_acc*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("Training complete! Model includes 3 anatomical measurements:")
    print("- Left subclavian diameter (mm)")
    print("- Aortic arch diameter (mm)")
    print("- Angle (degrees)")
    print("=" * 60)

if __name__ == "__main__":
    train_fast()