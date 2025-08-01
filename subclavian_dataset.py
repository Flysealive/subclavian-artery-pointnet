#!/usr/bin/env python3

import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import os
from pathlib import Path

class SubclavianDataset(data.Dataset):
    def __init__(self, 
                 numpy_dir, 
                 csv_file, 
                 npoints=2500, 
                 split='train', 
                 train_ratio=0.7,
                 val_ratio=0.15,
                 data_augmentation=True,
                 seed=42):
        """
        Dataset for Subclavian Artery Point Clouds
        
        Args:
            numpy_dir: Directory containing .npy files
            csv_file: CSV file with filename and label columns
            npoints: Number of points to sample from each point cloud
            split: 'train', 'val', or 'test'
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation (rest goes to test)
            data_augmentation: Whether to apply data augmentation
            seed: Random seed for reproducible splits
        """
        self.numpy_dir = Path(numpy_dir)
        self.npoints = npoints
        self.split = split
        self.data_augmentation = data_augmentation
        
        # Load labels from CSV
        df = pd.read_csv(csv_file)
        
        # Set random seed for reproducible splits
        np.random.seed(seed)
        
        # Create train/val/test splits
        indices = np.random.permutation(len(df))
        train_end = int(train_ratio * len(df))
        val_end = int((train_ratio + val_ratio) * len(df))
        
        if split == 'train':
            selected_indices = indices[:train_end]
        elif split == 'val':
            selected_indices = indices[train_end:val_end]
        elif split == 'test':
            selected_indices = indices[val_end:]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")
        
        # Filter dataframe based on split
        self.data = df.iloc[selected_indices].reset_index(drop=True)
        
        print(f"{split.upper()} set: {len(self.data)} samples")
        print(f"Class distribution: {self.data['label'].value_counts().to_dict()}")
    
    def __getitem__(self, index):
        # Get filename and label
        filename = self.data.iloc[index]['filename']
        label = self.data.iloc[index]['label']
        
        # Load point cloud from numpy file
        point_cloud_path = self.numpy_dir / filename
        point_cloud = np.load(point_cloud_path).astype(np.float32)
        
        # Handle different point cloud formats
        if point_cloud.shape[1] == 6:  # XYZ + normals
            point_set = point_cloud[:, :3]  # Use only XYZ coordinates
        elif point_cloud.shape[1] == 3:  # XYZ only
            point_set = point_cloud
        else:
            raise ValueError(f"Unexpected point cloud shape: {point_cloud.shape}")
        
        # Sample or pad to desired number of points
        if len(point_set) >= self.npoints:
            # Randomly sample npoints
            choice = np.random.choice(len(point_set), self.npoints, replace=False)
        else:
            # Pad with replacement if not enough points
            choice = np.random.choice(len(point_set), self.npoints, replace=True)
        
        point_set = point_set[choice, :]
        
        # Normalize point cloud
        # Center around origin
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)
        
        # Scale to unit sphere
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        if dist > 0:
            point_set = point_set / dist
        
        # Data augmentation
        if self.data_augmentation and self.split == 'train':
            # Random rotation around Y-axis
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
            point_set = point_set.dot(rotation_matrix)
            
            # Random jitter
            point_set += np.random.normal(0, 0.02, size=point_set.shape)
            
            # Random scaling
            scale = np.random.uniform(0.8, 1.2)
            point_set *= scale
        
        # Convert to torch tensors with correct data types
        point_set = torch.from_numpy(point_set.astype(np.float32))
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        
        return point_set, label
    
    def __len__(self):
        return len(self.data)

# Test the dataset
if __name__ == '__main__':
    # Test dataset creation
    dataset = SubclavianDataset(
        numpy_dir='numpy_arrays',
        csv_file='classification_labels.csv',
        split='train',
        npoints=1024
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    point_cloud, label = dataset[0]
    print(f"Point cloud shape: {point_cloud.shape}")
    print(f"Label: {label}")
    print(f"Point cloud type: {point_cloud.dtype}")
    print(f"Label type: {label.dtype}")