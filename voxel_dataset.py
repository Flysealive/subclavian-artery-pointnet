import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import random

class VoxelDataset(Dataset):
    """
    Dataset class for voxel data
    """
    def __init__(self, voxel_dir='voxel_arrays', labels_file='classification_labels.csv', 
                 mode='train', test_size=0.2, augment=True, random_seed=42):
        
        self.voxel_dir = voxel_dir
        self.augment = augment and (mode == 'train')
        
        labels_df = pd.read_csv(labels_file)
        
        labels_df['filename'] = labels_df['filename'].str.replace('.npy', '')
        
        self.data = []
        self.labels = []
        
        for _, row in labels_df.iterrows():
            voxel_file = os.path.join(voxel_dir, row['filename'] + '.npy')
            if os.path.exists(voxel_file):
                self.data.append(voxel_file)
                self.labels.append(row['label'])
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.labels, test_size=test_size, 
            random_state=random_seed, stratify=self.labels
        )
        
        if mode == 'train':
            self.data = X_train
            self.labels = y_train
        else:
            self.data = X_test
            self.labels = y_test
        
        print(f"{mode} dataset: {len(self.data)} samples")
        print(f"Class distribution: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        voxel_path = self.data[idx]
        voxel = np.load(voxel_path).astype(np.float32)
        label = self.labels[idx]
        
        if self.augment:
            voxel = self.augment_voxel(voxel)
        
        return torch.FloatTensor(voxel), torch.LongTensor([label])[0]
    
    def augment_voxel(self, voxel):
        """
        Apply data augmentation to voxel
        """
        if random.random() > 0.5:
            axis = random.randint(0, 2)
            voxel = np.flip(voxel, axis=axis).copy()
        
        if random.random() > 0.5:
            k = random.randint(1, 3)
            axes = random.sample([0, 1, 2], 2)
            voxel = np.rot90(voxel, k=k, axes=axes).copy()
        
        if random.random() > 0.3:
            noise = np.random.normal(0, 0.01, voxel.shape)
            voxel = voxel + noise * voxel
        
        if random.random() > 0.5:
            scale = random.uniform(0.9, 1.1)
            voxel = voxel * scale
        
        return voxel

class BalancedVoxelDataset(VoxelDataset):
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
            
            for class_label in unique:
                class_indices = np.where(self.labels == class_label)[0]
                class_data = self.data[class_indices]
                
                if len(class_indices) < max_count:
                    oversample_indices = np.random.choice(
                        class_indices, 
                        size=max_count - len(class_indices), 
                        replace=True
                    )
                    class_data = np.concatenate([
                        self.data[class_indices],
                        self.data[oversample_indices]
                    ])
                
                balanced_data.extend(class_data)
                balanced_labels.extend([class_label] * max_count)
            
            self.data = np.array(balanced_data)
            self.labels = np.array(balanced_labels)
            
            print(f"Balanced dataset: {len(self.data)} samples")
            print(f"Class distribution: {np.bincount(self.labels)}")

def get_voxel_dataloaders(batch_size=8, voxel_dir='voxel_arrays', 
                         labels_file='classification_labels.csv', 
                         balanced=True, num_workers=0):
    """
    Create train and test dataloaders for voxel data
    """
    DatasetClass = BalancedVoxelDataset if balanced else VoxelDataset
    
    train_dataset = DatasetClass(
        voxel_dir=voxel_dir,
        labels_file=labels_file,
        mode='train',
        augment=True
    )
    
    test_dataset = DatasetClass(
        voxel_dir=voxel_dir,
        labels_file=labels_file,
        mode='test',
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
    
    return train_loader, test_loader