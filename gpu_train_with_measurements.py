#!/usr/bin/env python3
"""
GPU-optimized training script for PointNet with anatomical measurements
Supports both CPU and GPU training
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

class SimpleDatasetWithMeasurements(Dataset):
    def __init__(self, csv_file, numpy_dir='numpy_arrays', split='train', npoints=1024):
        self.numpy_dir = numpy_dir
        self.npoints = npoints
        
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
        
        # Normalize measurements
        self.mean_measurements = {
            'left_subclavian': df['left_subclavian'].mean(),
            'aortic_arch': df['aortic_arch'].mean(),
            'angle': df['angle'].mean()
        }
        self.std_measurements = {
            'left_subclavian': df['left_subclavian'].std() + 1e-6,
            'aortic_arch': df['aortic_arch'].std() + 1e-6,
            'angle': df['angle'].std() + 1e-6
        }
        
        print(f"{split.upper()} set: {len(self.data)} samples")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load point cloud
        file_path = os.path.join(self.numpy_dir, row['filename'])
        points = np.load(file_path).astype(np.float32)
        
        # Sample points
        if points.shape[0] > self.npoints:
            choice = np.random.choice(points.shape[0], self.npoints, replace=False)
        else:
            choice = np.random.choice(points.shape[0], self.npoints, replace=True)
        points = points[choice, :]
        
        # Center and normalize point cloud
        points = points - points.mean(axis=0)
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
        
        # Data augmentation for training
        if hasattr(self, 'augment') and self.augment:
            # Random rotation around Y axis
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
            points = points @ rotation_matrix
            
            # Random jitter
            points += np.random.normal(0, 0.02, size=points.shape)
        
        # Get measurements and normalize
        measurements = np.array([
            (row['left_subclavian'] - self.mean_measurements['left_subclavian']) / self.std_measurements['left_subclavian'],
            (row['aortic_arch'] - self.mean_measurements['aortic_arch']) / self.std_measurements['aortic_arch'],
            (row['angle'] - self.mean_measurements['angle']) / self.std_measurements['angle']
        ], dtype=np.float32)
        
        label = int(row['label'])
        
        return torch.from_numpy(points), torch.from_numpy(measurements), label

class EnhancedPointNetWithMeasurements(nn.Module):
    def __init__(self, num_classes=2, num_measurements=3):
        super().__init__()
        
        # Point cloud feature extraction with more layers
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
        
        # Measurement processing network
        self.meas_fc1 = nn.Linear(num_measurements, 32)
        self.meas_fc2 = nn.Linear(32, 64)
        self.meas_bn1 = nn.BatchNorm1d(32)
        self.meas_bn2 = nn.BatchNorm1d(64)
        
        # Combined classifier
        self.fc1 = nn.Linear(1024 + 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.bn_fc3 = nn.BatchNorm1d(128)
        
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, points, measurements):
        # points: (B, N, 3)
        # measurements: (B, 3)
        
        B, N, _ = points.size()
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
        
        # Combine features
        combined = torch.cat([x, m], dim=1)  # (B, 1024+64)
        
        # Classification
        x = F.relu(self.bn_fc1(self.fc1(combined)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        
        return F.log_softmax(x, dim=1)

def train():
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("GPU/CUDA Configuration Check")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print("[OK] GPU training enabled!")
    else:
        print("\n[WARNING] GPU NOT AVAILABLE - Training will use CPU (slower)")
        print("\nTo enable GPU training on Windows with NVIDIA 4060Ti:")
        print("1. Uninstall current PyTorch: pip uninstall torch torchvision -y")
        print("2. Install CUDA-enabled PyTorch:")
        print("   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("3. Make sure NVIDIA drivers are up to date")
        print("4. Restart this script")
    
    print("=" * 60)
    print(f"\nUsing device: {device}")
    
    # Create datasets with augmentation for training
    train_dataset = SimpleDatasetWithMeasurements('classification_labels_with_measurements.csv', split='train')
    train_dataset.augment = True  # Enable augmentation
    val_dataset = SimpleDatasetWithMeasurements('classification_labels_with_measurements.csv', split='val')
    test_dataset = SimpleDatasetWithMeasurements('classification_labels_with_measurements.csv', split='test')
    
    # Create dataloaders with optimized settings
    batch_size = 16 if torch.cuda.is_available() else 8
    num_workers = 4 if torch.cuda.is_available() else 0
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                            drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
    # Create model
    model = EnhancedPointNetWithMeasurements().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Training
    print("\nStarting training...")
    best_val_acc = 0
    epochs = 50
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (points, measurements, labels) in enumerate(train_loader):
            points, measurements, labels = points.to(device), measurements.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output = model(points, measurements)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(labels).sum().item()
            train_total += labels.size(0)
            
            # Print progress
            if torch.cuda.is_available() and batch_idx % 5 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}', end='\r')
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_class_correct = [0, 0]
        val_class_total = [0, 0]
        
        with torch.no_grad():
            for points, measurements, labels in val_loader:
                points, measurements, labels = points.to(device), measurements.to(device), labels.to(device)
                
                output = model(points, measurements)
                pred = output.argmax(dim=1)
                
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    prediction = pred[i].item()
                    
                    val_class_total[label] += 1
                    val_total += 1
                    
                    if prediction == label:
                        val_class_correct[label] += 1
                        val_correct += 1
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        val_class0_acc = 100. * val_class_correct[0] / val_class_total[0] if val_class_total[0] > 0 else 0
        val_class1_acc = 100. * val_class_correct[1] / val_class_total[1] if val_class_total[1] > 0 else 0
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1:3d}/{epochs}: Train={train_acc:.1f}%, Val={val_acc:.1f}% '
              f'(C0={val_class0_acc:.1f}%, C1={val_class1_acc:.1f}%) '
              f'LR={scheduler.get_last_lr()[0]:.5f} Time={epoch_time:.1f}s')
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('cls', exist_ok=True)
            torch.save(model.state_dict(), 'cls/gpu_with_measurements.pth')
            print(f'  -> Saved best model (Val Acc: {val_acc:.1f}%)')
    
    # Test
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    
    model.load_state_dict(torch.load('cls/gpu_with_measurements.pth', map_location=device))
    model.eval()
    
    test_correct = 0
    test_total = 0
    class_correct = [0, 0]
    class_total = [0, 0]
    confusion_matrix = torch.zeros(2, 2)
    
    with torch.no_grad():
        for points, measurements, labels in test_loader:
            points, measurements, labels = points.to(device), measurements.to(device), labels.to(device)
            
            output = model(points, measurements)
            pred = output.argmax(dim=1)
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                prediction = pred[i].item()
                
                confusion_matrix[label, prediction] += 1
                class_total[label] += 1
                test_total += 1
                
                if prediction == label:
                    class_correct[label] += 1
                    test_correct += 1
    
    print(f"\nOverall Test Accuracy: {100.*test_correct/test_total:.1f}% ({test_correct}/{test_total})")
    
    for i in range(2):
        if class_total[i] > 0:
            acc = 100.*class_correct[i]/class_total[i]
            print(f"Class {i} Accuracy: {acc:.1f}% ({class_correct[i]}/{class_total[i]})")
    
    print("\nConfusion Matrix:")
    print("       Pred 0  Pred 1")
    print(f"True 0:  {int(confusion_matrix[0,0]):4d}   {int(confusion_matrix[0,1]):4d}")
    print(f"True 1:  {int(confusion_matrix[1,0]):4d}   {int(confusion_matrix[1,1]):4d}")
    
    if torch.cuda.is_available():
        print(f"\n[OK] Training completed on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\n[OK] Training completed on CPU")

if __name__ == "__main__":
    train()