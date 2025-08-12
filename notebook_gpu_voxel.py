"""
Jupyter Notebook GPU Voxel Training
====================================
Run this in Jupyter notebook cells for interactive GPU training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import time
from tqdm.notebook import tqdm

# Check GPU availability
def check_cuda():
    if torch.cuda.is_available():
        print(f"✓ CUDA Available: {torch.cuda.is_available()}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
        
        # Set optimal GPU settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        print("✓ CUDNN optimization enabled")
        return True
    else:
        print("✗ CUDA not available, using CPU")
        return False

# Simple function to train on GPU
def quick_gpu_train(model, train_loader, val_loader, epochs=50, device='cuda'):
    """
    Quick training function optimized for Jupyter notebooks
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()  # For mixed precision
    
    train_losses = []
    val_accs = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for data, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            with autocast():
                outputs = model(data)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        val_accs.append(val_acc)
        
        # Live plotting in notebook
        if (epoch + 1) % 5 == 0:
            clear_output(wait=True)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.plot(train_losses)
            ax1.set_title('Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True)
            
            ax2.plot(val_accs)
            ax2.set_title('Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.grid(True)
            
            plt.suptitle(f'Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | Acc: {val_acc:.4f}')
            plt.tight_layout()
            display(plt.gcf())
            plt.close()
    
    print(f"\n✓ Training complete! Final accuracy: {val_accs[-1]:.4f}")
    return model, train_losses, val_accs

# Example usage in notebook cells:
"""
# Cell 1: Setup
%matplotlib inline
import sys
sys.path.append('.')
from notebook_gpu_voxel import *
from voxel_cnn_model import VoxelCNN
from voxel_dataset import get_voxel_dataloaders

# Cell 2: Check GPU
gpu_available = check_cuda()
device = 'cuda' if gpu_available else 'cpu'

# Cell 3: Prepare data
!python stl_to_voxel.py  # Convert STL to voxels first
train_loader, val_loader = get_voxel_dataloaders(batch_size=16, balanced=True)

# Cell 4: Create and train model
model = VoxelCNN(num_classes=2)
trained_model, losses, accs = quick_gpu_train(
    model, train_loader, val_loader, 
    epochs=50, device=device
)

# Cell 5: Save model
torch.save(trained_model.state_dict(), 'voxel_model_gpu.pth')
"""

# Memory-efficient batch processing for large datasets
def process_large_dataset_gpu(model, data_loader, device='cuda'):
    """
    Process large datasets efficiently on GPU
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Processing'):
            data, _ = batch
            data = data.to(device)
            
            # Use mixed precision for inference
            with autocast():
                outputs = model(data)
                probs = torch.softmax(outputs, dim=1)
            
            predictions = torch.argmax(probs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
            
            # Clear GPU cache periodically
            if len(all_predictions) % 100 == 0:
                torch.cuda.empty_cache()
    
    return np.array(all_predictions), np.array(all_probabilities)

# GPU memory monitoring
def monitor_gpu_memory():
    """
    Monitor GPU memory usage
    """
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        print(f"Cached: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
        print(f"Max Allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
        
        # Clear cache if needed
        torch.cuda.empty_cache()
        print("Cache cleared!")

# Optimized data loading for GPU
def get_gpu_optimized_loaders(batch_size=16, num_workers=4):
    """
    Create data loaders optimized for GPU training
    """
    from voxel_dataset import BalancedVoxelDataset
    from torch.utils.data import DataLoader
    
    train_dataset = BalancedVoxelDataset(
        voxel_dir='voxel_arrays',
        labels_file='classification_labels.csv',
        mode='train',
        augment=True
    )
    
    val_dataset = BalancedVoxelDataset(
        voxel_dir='voxel_arrays',
        labels_file='classification_labels.csv',
        mode='test',
        augment=False
    )
    
    # GPU-optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Important for GPU
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    print("GPU Voxel Training Module for Jupyter Notebooks")
    print("="*50)
    check_cuda()
    print("\nImport this module in your notebook:")
    print("from notebook_gpu_voxel import *")