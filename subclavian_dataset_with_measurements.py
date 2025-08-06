#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class SubclavianDatasetWithMeasurements(Dataset):
    """
    Dataset for subclavian artery classification with anatomical measurements
    """
    def __init__(self, numpy_dir, csv_file, split='train', npoints=1024, 
                 train_ratio=0.7, val_ratio=0.15, data_augmentation=False,
                 normalize_measurements=True):
        """
        Args:
            numpy_dir: Directory containing .npy point cloud files
            csv_file: CSV with columns: filename, label, left_subclavian_diameter, 
                     aortic_arch_diameter, angle
            split: 'train', 'val', or 'test'
            npoints: Number of points to sample
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            data_augmentation: Apply augmentation to training data
            normalize_measurements: Whether to normalize anatomical measurements
        """
        self.numpy_dir = numpy_dir
        self.npoints = npoints
        self.data_augmentation = data_augmentation
        self.normalize_measurements = normalize_measurements
        
        # Load CSV with measurements
        df = pd.read_csv(csv_file)
        
        # Ensure required columns exist
        required_cols = ['filename', 'label', 'left_subclavian_diameter', 
                        'aortic_arch_diameter', 'angle']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Filter for existing files
        existing_files = []
        existing_labels = []
        existing_measurements = []
        
        for idx, row in df.iterrows():
            file_path = os.path.join(numpy_dir, row['filename'])
            if os.path.exists(file_path):
                existing_files.append(row['filename'])
                existing_labels.append(row['label'])
                existing_measurements.append([
                    row['left_subclavian_diameter'],
                    row['aortic_arch_diameter'],
                    row['angle']
                ])
        
        # Split data
        n_samples = len(existing_files)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        indices = np.random.permutation(n_samples)
        
        if split == 'train':
            selected_indices = indices[:n_train]
        elif split == 'val':
            selected_indices = indices[n_train:n_train + n_val]
        else:  # test
            selected_indices = indices[n_train + n_val:]
        
        self.files = [existing_files[i] for i in selected_indices]
        self.labels = [existing_labels[i] for i in selected_indices]
        self.measurements = np.array([existing_measurements[i] for i in selected_indices])
        
        # Calculate normalization statistics from training set
        if split == 'train' and normalize_measurements:
            self.meas_mean = np.mean(self.measurements, axis=0)
            self.meas_std = np.std(self.measurements, axis=0) + 1e-6
            # Save normalization stats for val/test sets
            np.save('measurement_stats.npy', {'mean': self.meas_mean, 'std': self.meas_std})
        elif normalize_measurements and os.path.exists('measurement_stats.npy'):
            # Load normalization stats for val/test sets
            stats = np.load('measurement_stats.npy', allow_pickle=True).item()
            self.meas_mean = stats['mean']
            self.meas_std = stats['std']
        else:
            self.meas_mean = np.zeros(3)
            self.meas_std = np.ones(3)
        
        print(f"{split.upper()} set: {len(self.files)} samples")
        print(f"Class distribution: {dict(zip(*np.unique(self.labels, return_counts=True)))}")
        print(f"Measurement ranges:")
        print(f"  Left subclavian diameter: {self.measurements[:, 0].min():.2f} - {self.measurements[:, 0].max():.2f} mm")
        print(f"  Aortic arch diameter: {self.measurements[:, 1].min():.2f} - {self.measurements[:, 1].max():.2f} mm")
        print(f"  Angle: {self.measurements[:, 2].min():.2f} - {self.measurements[:, 2].max():.2f} degrees")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load point cloud
        file_path = os.path.join(self.numpy_dir, self.files[idx])
        points = np.load(file_path).astype(np.float32)
        
        # Sample points
        if points.shape[0] > self.npoints:
            choice = np.random.choice(points.shape[0], self.npoints, replace=False)
        else:
            choice = np.random.choice(points.shape[0], self.npoints, replace=True)
        
        points = points[choice, :]
        
        # Center and normalize point cloud
        points = points - np.mean(points, axis=0)
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
        
        # Data augmentation for training
        if self.data_augmentation and np.random.random() > 0.5:
            # Random rotation around Y axis
            theta = np.random.uniform(0, 2*np.pi)
            rotation_matrix = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
            points = points @ rotation_matrix.T
            
            # Random jitter
            points += np.random.normal(0, 0.02, size=points.shape)
            
            # Random scale
            scale = np.random.uniform(0.8, 1.2)
            points *= scale
        
        # Get measurements and normalize
        measurements = self.measurements[idx].copy()
        if self.normalize_measurements:
            measurements = (measurements - self.meas_mean) / self.meas_std
        
        # Convert to tensors
        points = torch.from_numpy(points.T).float()  # Shape: (3, npoints)
        measurements = torch.from_numpy(measurements).float()  # Shape: (3,)
        label = torch.tensor([self.labels[idx]], dtype=torch.long)
        
        return points, measurements, label


def create_sample_csv_with_measurements(original_csv, output_csv):
    """
    Create a sample CSV with random anatomical measurements for testing
    """
    df = pd.read_csv(original_csv)
    
    # Add random measurements (replace with real data)
    np.random.seed(42)
    n_samples = len(df)
    
    # Generate realistic random measurements
    # Adjust ranges based on your actual data
    df['left_subclavian_diameter'] = np.random.uniform(5, 15, n_samples)  # mm
    df['aortic_arch_diameter'] = np.random.uniform(20, 40, n_samples)  # mm
    df['angle'] = np.random.uniform(30, 150, n_samples)  # degrees
    
    # Add some correlation with labels (optional, for testing)
    # Make class 1 tend to have larger measurements
    class_1_mask = df['label'] == 1
    df.loc[class_1_mask, 'left_subclavian_diameter'] *= 1.2
    df.loc[class_1_mask, 'aortic_arch_diameter'] *= 1.15
    df.loc[class_1_mask, 'angle'] *= 0.9
    
    df.to_csv(output_csv, index=False)
    print(f"Created sample CSV with measurements: {output_csv}")
    return df


if __name__ == "__main__":
    # Test the dataset
    import sys
    
    # Create sample CSV with measurements for testing
    if not os.path.exists('classification_labels_with_measurements.csv'):
        create_sample_csv_with_measurements(
            'classification_labels.csv',
            'classification_labels_with_measurements.csv'
        )
    
    # Test dataset loading
    dataset = SubclavianDatasetWithMeasurements(
        numpy_dir='numpy_arrays',
        csv_file='classification_labels_with_measurements.csv',
        split='train',
        npoints=1024,
        data_augmentation=True
    )
    
    # Get a sample
    points, measurements, label = dataset[0]
    print(f"\nSample data shapes:")
    print(f"Points: {points.shape}")
    print(f"Measurements: {measurements.shape}")
    print(f"Label: {label.shape}")
    print(f"Measurements values: {measurements}")