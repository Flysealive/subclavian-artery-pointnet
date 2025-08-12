"""
Voxel Dataset with Anatomical Measurements
===========================================
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

class VoxelMeasurementDataset(Dataset):
    """
    Dataset that combines voxel data with anatomical measurements
    """
    def __init__(self, voxel_dir='voxel_arrays', 
                 labels_file='classification_labels_with_measurements.csv',
                 mode='train', test_size=0.2, augment=True, 
                 normalize_measurements=True, random_seed=42):
        
        self.voxel_dir = voxel_dir
        self.augment = augment and (mode == 'train')
        self.normalize_measurements = normalize_measurements
        
        # Load labels with measurements
        labels_df = pd.read_csv(labels_file)
        
        # Extract filename without extension
        labels_df['filename'] = labels_df['filename'].str.replace('.npy', '')
        
        # Prepare data lists
        self.data = []
        self.labels = []
        self.measurements = []
        
        # Measurement columns
        measurement_cols = ['left_subclavian_diameter_mm', 
                          'aortic_arch_diameter_mm', 
                          'angle_degrees']
        
        for _, row in labels_df.iterrows():
            voxel_file = os.path.join(voxel_dir, row['filename'] + '.npy')
            if os.path.exists(voxel_file):
                self.data.append(voxel_file)
                self.labels.append(row['label'])
                self.measurements.append(row[measurement_cols].values)
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.measurements = np.array(self.measurements, dtype=np.float32)
        
        # Split data
        X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
            self.data, self.labels, self.measurements,
            test_size=test_size, random_state=random_seed, 
            stratify=self.labels
        )
        
        if mode == 'train':
            self.data = X_train
            self.labels = y_train
            self.measurements = m_train
        else:
            self.data = X_test
            self.labels = y_test
            self.measurements = m_test
        
        # Normalize measurements
        if self.normalize_measurements:
            self.scaler = StandardScaler()
            if mode == 'train':
                self.measurements = self.scaler.fit_transform(self.measurements)
                # Save scaler parameters for test set
                self.mean_ = self.scaler.mean_
                self.scale_ = self.scaler.scale_
            else:
                # Use training set statistics for normalization
                if hasattr(self, 'mean_') and hasattr(self, 'scale_'):
                    self.measurements = (self.measurements - self.mean_) / self.scale_
                else:
                    # If no training stats available, fit on test set
                    self.measurements = self.scaler.fit_transform(self.measurements)
        
        print(f"{mode} dataset: {len(self.data)} samples")
        print(f"Class distribution: {np.bincount(self.labels)}")
        print(f"Measurement shape: {self.measurements.shape}")
        
        # Print measurement statistics
        if mode == 'train':
            print(f"\nMeasurement statistics (normalized={normalize_measurements}):")
            for i, col in enumerate(measurement_cols):
                values = self.measurements[:, i]
                print(f"  {col}: mean={values.mean():.3f}, std={values.std():.3f}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Load voxel data
        voxel_path = self.data[idx]
        voxel = np.load(voxel_path).astype(np.float32)
        
        # Get label and measurements
        label = self.labels[idx]
        measurements = self.measurements[idx].astype(np.float32)
        
        # Apply augmentation to voxel
        if self.augment:
            voxel = self.augment_voxel(voxel)
            # Optionally augment measurements slightly
            measurements = self.augment_measurements(measurements)
        
        return (torch.FloatTensor(voxel), 
                torch.FloatTensor(measurements), 
                torch.LongTensor([label])[0])
    
    def augment_voxel(self, voxel):
        """Apply data augmentation to voxel"""
        # Random flip
        if random.random() > 0.5:
            axis = random.randint(0, 2)
            voxel = np.flip(voxel, axis=axis).copy()
        
        # Random rotation
        if random.random() > 0.5:
            k = random.randint(1, 3)
            axes = random.sample([0, 1, 2], 2)
            voxel = np.rot90(voxel, k=k, axes=axes).copy()
        
        # Add noise
        if random.random() > 0.3:
            noise = np.random.normal(0, 0.01, voxel.shape)
            voxel = voxel + noise * voxel
        
        # Random scaling
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            voxel = voxel * scale
        
        return voxel
    
    def augment_measurements(self, measurements):
        """Apply slight augmentation to measurements"""
        if random.random() > 0.7:
            # Add small noise to measurements (±2% variation)
            noise = np.random.normal(0, 0.02, measurements.shape)
            measurements = measurements + noise
        return measurements


class BalancedVoxelMeasurementDataset(VoxelMeasurementDataset):
    """
    Balanced dataset with oversampling for minority class
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if hasattr(self, 'labels'):
            unique, counts = np.unique(self.labels, return_counts=True)
            max_count = counts.max()
            
            balanced_data = []
            balanced_labels = []
            balanced_measurements = []
            
            for class_label in unique:
                class_indices = np.where(self.labels == class_label)[0]
                
                if len(class_indices) < max_count:
                    # Oversample minority class
                    oversample_indices = np.random.choice(
                        class_indices, 
                        size=max_count - len(class_indices), 
                        replace=True
                    )
                    all_indices = np.concatenate([class_indices, oversample_indices])
                else:
                    all_indices = class_indices
                
                balanced_data.extend(self.data[all_indices])
                balanced_labels.extend([class_label] * max_count)
                balanced_measurements.extend(self.measurements[all_indices])
            
            self.data = np.array(balanced_data)
            self.labels = np.array(balanced_labels)
            self.measurements = np.array(balanced_measurements)
            
            print(f"\nBalanced dataset: {len(self.data)} samples")
            print(f"Class distribution: {np.bincount(self.labels)}")


def get_measurement_dataloaders(batch_size=8, voxel_dir='voxel_arrays',
                               labels_file='classification_labels_with_measurements.csv',
                               balanced=True, num_workers=0):
    """
    Create train and test dataloaders with measurements
    """
    DatasetClass = BalancedVoxelMeasurementDataset if balanced else VoxelMeasurementDataset
    
    # Create train dataset (this will fit the scaler)
    train_dataset = DatasetClass(
        voxel_dir=voxel_dir,
        labels_file=labels_file,
        mode='train',
        augment=True,
        normalize_measurements=True
    )
    
    # Create test dataset (will use train scaler parameters)
    test_dataset = VoxelMeasurementDataset(  # Don't balance test set
        voxel_dir=voxel_dir,
        labels_file=labels_file,
        mode='test',
        augment=False,
        normalize_measurements=True
    )
    
    # Copy scaler parameters from train to test
    if hasattr(train_dataset, 'mean_'):
        test_dataset.mean_ = train_dataset.mean_
        test_dataset.scale_ = train_dataset.scale_
        test_dataset.measurements = (test_dataset.measurements - test_dataset.mean_) / test_dataset.scale_
    
    # Custom collate function for multiple inputs
    def collate_fn(batch):
        voxels = torch.stack([item[0] for item in batch])
        measurements = torch.stack([item[1] for item in batch])
        labels = torch.stack([torch.tensor(item[2]) for item in batch])
        return voxels, measurements, labels
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )
    
    return train_loader, test_loader