#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from pathlib import Path

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
        points = points / max_dist
        
        # Get measurements and normalize
        measurements = np.array([
            (row['left_subclavian'] - self.mean_measurements['left_subclavian']) / self.std_measurements['left_subclavian'],
            (row['aortic_arch'] - self.mean_measurements['aortic_arch']) / self.std_measurements['aortic_arch'],
            (row['angle'] - self.mean_measurements['angle']) / self.std_measurements['angle']
        ], dtype=np.float32)
        
        label = int(row['label'])
        
        return torch.from_numpy(points), torch.from_numpy(measurements), label

class SimplePointNetWithMeasurements(nn.Module):
    def __init__(self, num_classes=2, num_measurements=3):
        super().__init__()
        
        # Point cloud feature extraction
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        
        # Combined classifier
        self.fc1 = nn.Linear(512 + num_measurements, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.3)
        
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
        
        # Global max pooling
        x = torch.max(x, 2)[0]  # (B, 512)
        
        # Combine with measurements
        x = torch.cat([x, measurements], dim=1)  # (B, 512+3)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = SimpleDatasetWithMeasurements('classification_labels_with_measurements.csv', split='train')
    val_dataset = SimpleDatasetWithMeasurements('classification_labels_with_measurements.csv', split='val')
    test_dataset = SimpleDatasetWithMeasurements('classification_labels_with_measurements.csv', split='test')
    
    # Create dataloaders with larger batch size to avoid batch norm issues
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Create model
    model = SimplePointNetWithMeasurements().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    print("\nStarting training...")
    best_val_acc = 0
    
    for epoch in range(30):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for points, measurements, labels in train_loader:
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
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for points, measurements, labels in val_loader:
                points, measurements, labels = points.to(device), measurements.to(device), labels.to(device)
                
                output = model(points, measurements)
                pred = output.argmax(dim=1)
                val_correct += pred.eq(labels).sum().item()
                val_total += labels.size(0)
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1:3d}: Train Acc={train_acc:.1f}%, Val Acc={val_acc:.1f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('cls', exist_ok=True)
            torch.save(model.state_dict(), 'cls/simple_with_measurements.pth')
    
    # Test
    print("\n=== TEST RESULTS ===")
    model.load_state_dict(torch.load('cls/simple_with_measurements.pth'))
    model.eval()
    
    test_correct = 0
    test_total = 0
    class_correct = [0, 0]
    class_total = [0, 0]
    
    with torch.no_grad():
        for points, measurements, labels in test_loader:
            points, measurements, labels = points.to(device), measurements.to(device), labels.to(device)
            
            output = model(points, measurements)
            pred = output.argmax(dim=1)
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                prediction = pred[i].item()
                
                class_total[label] += 1
                test_total += 1
                
                if prediction == label:
                    class_correct[label] += 1
                    test_correct += 1
    
    print(f"Overall Accuracy: {100.*test_correct/test_total:.1f}%")
    for i in range(2):
        if class_total[i] > 0:
            print(f"Class {i} Accuracy: {100.*class_correct[i]/class_total[i]:.1f}% ({class_correct[i]}/{class_total[i]})")

if __name__ == "__main__":
    train()