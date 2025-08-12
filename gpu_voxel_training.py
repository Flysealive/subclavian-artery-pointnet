"""
GPU-Optimized Voxel Training Script with CUDA Acceleration
===========================================================
This script is optimized for NVIDIA GPU training with automatic mixed precision
and other GPU-specific optimizations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
from datetime import datetime
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

from voxel_cnn_model import VoxelCNN, VoxelResNet
from voxel_dataset import get_voxel_dataloaders

class GPUVoxelTrainer:
    def __init__(self, model, device, model_name="voxel_cnn", use_amp=True):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.use_amp = use_amp and torch.cuda.is_available()
        
        if self.use_amp:
            self.scaler = GradScaler()
            print("✓ Using Automatic Mixed Precision (AMP) for faster training")
        
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_precision': [], 'val_recall': [], 'val_f1': [],
            'learning_rates': [], 'gpu_memory': []
        }
        
        # Enable cudnn benchmarking for faster training
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            print("✓ CUDNN benchmarking enabled")
    
    def get_gpu_memory(self):
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**3  # GB
        return 0
    
    def train_epoch(self, train_loader, criterion, optimizer, scheduler=None):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc="Training", ncols=100)
        
        for batch_idx, (data, labels) in enumerate(progress_bar):
            data, labels = data.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Use automatic mixed precision for faster training
            if self.use_amp:
                with autocast():
                    outputs = self.model(data)
                    loss = criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            
            if scheduler is not None and hasattr(scheduler, 'step'):
                if isinstance(scheduler, OneCycleLR):
                    scheduler.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'gpu_mem': f'{self.get_gpu_memory():.1f}GB'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc="Validation", ncols=100):
                data, labels = data.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(data)
                        loss = criterion(outputs, labels)
                else:
                    outputs = self.model(data)
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        metrics = self.calculate_metrics(all_labels, all_preds)
        
        return epoch_loss, metrics, all_labels, all_preds
    
    def calculate_metrics(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, patience=20, use_scheduler='onecycle'):
        criterion = nn.CrossEntropyLoss()
        
        # Use AdamW optimizer for better regularization
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Setup scheduler
        if use_scheduler == 'onecycle':
            scheduler = OneCycleLR(
                optimizer, 
                max_lr=lr*10,
                epochs=epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,
                anneal_strategy='cos'
            )
            print("✓ Using OneCycle learning rate scheduler")
        else:
            scheduler = None
        
        best_val_acc = 0
        best_epoch = 0
        patience_counter = 0
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Training {self.model_name} on {self.device}")
        print(f"Epochs: {epochs} | Batch Size: {train_loader.batch_size}")
        print(f"Learning Rate: {lr} | Patience: {patience}")
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB" if torch.cuda.is_available() else "")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 40)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer, scheduler)
            
            # Validation
            val_loss, val_metrics, val_labels, val_preds = self.validate(val_loader, criterion)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['learning_rates'].append(current_lr)
            self.history['gpu_memory'].append(self.get_gpu_memory())
            
            # Print metrics
            epoch_time = time.time() - epoch_start
            print(f"Time: {epoch_time:.1f}s | LR: {current_lr:.6f}")
            print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val   - Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")
            print(f"GPU Memory: {self.get_gpu_memory():.2f}GB")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_epoch = epoch + 1
                patience_counter = 0
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'history': self.history
                }
                
                if self.use_amp:
                    checkpoint['scaler_state_dict'] = self.scaler.state_dict()
                
                torch.save(checkpoint, f'best_{self.model_name}_gpu.pth')
                print(f"✓ New best model saved! Val Acc: {best_val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n✗ Early stopping triggered at epoch {epoch+1}")
                break
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Total Time: {total_time/60:.1f} minutes")
        print(f"Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
        print(f"Final GPU Memory: {self.get_gpu_memory():.2f}GB")
        print(f"{'='*60}\n")
        
        # Generate final confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        self.plot_confusion_matrix(cm)
        
        return self.history
    
    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Class 0', 'Class 1'],
                   yticklabels=['Class 0', 'Class 1'],
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {self.model_name} (GPU)', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add accuracy text
        accuracy = np.trace(cm) / np.sum(cm)
        plt.text(0.5, -0.1, f'Accuracy: {accuracy:.2%}', 
                ha='center', transform=plt.gca().transAxes, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{self.model_name}_gpu_confusion_matrix.png', dpi=150)
        plt.close()

def check_gpu_status():
    """Check and display GPU status"""
    print("\n" + "="*60)
    print("GPU STATUS CHECK")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available!")
        print(f"  - PyTorch version: {torch.__version__}")
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    - Memory: {props.total_memory/1024**3:.1f}GB")
            print(f"    - Compute Capability: {props.major}.{props.minor}")
            print(f"    - Multi-processors: {props.multi_processor_count}")
        
        # Test GPU
        print("\n  Testing GPU computation...")
        test_tensor = torch.randn(2, 1, 64, 64, 64).cuda()
        print(f"  ✓ Successfully created tensor on GPU")
        print(f"  ✓ Tensor shape: {test_tensor.shape}")
        print(f"  ✓ Tensor device: {test_tensor.device}")
        
        return True
    else:
        print("✗ CUDA is NOT available")
        print("  Running on CPU (will be slower)")
        print("\n  To use GPU:")
        print("  1. Ensure NVIDIA GPU is installed")
        print("  2. Install CUDA-enabled PyTorch:")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False

def main():
    parser = argparse.ArgumentParser(description='GPU-Optimized Voxel CNN Training')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)  # Increased for GPU
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--voxel_size', type=int, default=64)
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--no_amp', action='store_true', help='Disable AMP')
    parser.add_argument('--num_workers', type=int, default=4)  # For faster data loading
    
    args = parser.parse_args()
    
    # Check GPU status
    gpu_available = check_gpu_status()
    
    # Set device
    device = torch.device('cuda' if gpu_available else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Convert STL to voxels if needed
    if not os.path.exists('voxel_arrays'):
        print("\nVoxel arrays not found. Running conversion...")
        from stl_to_voxel import convert_dataset_to_voxels
        convert_dataset_to_voxels(voxel_size=args.voxel_size)
    
    # Load data with multiple workers for faster loading
    print(f"\nLoading data with {args.num_workers} workers...")
    train_loader, test_loader = get_voxel_dataloaders(
        batch_size=args.batch_size,
        balanced=args.balanced,
        num_workers=args.num_workers
    )
    
    # Create model
    if args.model == 'resnet':
        model = VoxelResNet(num_classes=2, voxel_size=args.voxel_size)
        model_name = "voxel_resnet"
    else:
        model = VoxelCNN(num_classes=2, voxel_size=args.voxel_size)
        model_name = "voxel_cnn"
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.1f}MB")
    
    # Train model
    trainer = GPUVoxelTrainer(model, device, model_name, use_amp=not args.no_amp)
    history = trainer.train(
        train_loader, test_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience
    )
    
    # Save history
    with open(f'{model_name}_gpu_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    # Plot detailed results
    plot_training_results(history, model_name)
    
    print("\n✓ Training complete! Check the generated plots and model files.")

def plot_training_results(history, model_name):
    """Create comprehensive training plots"""
    fig = plt.figure(figsize=(20, 12))
    
    # Loss plot
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(history['train_acc'], label='Train Acc', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Metrics plot
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(history['val_precision'], label='Precision', linewidth=2)
    ax3.plot(history['val_recall'], label='Recall', linewidth=2)
    ax3.plot(history['val_f1'], label='F1 Score', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.set_title('Validation Metrics')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Learning rate plot
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(history['learning_rates'], linewidth=2, color='orange')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # GPU memory plot
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(history['gpu_memory'], linewidth=2, color='green')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('GPU Memory (GB)')
    ax5.set_title('GPU Memory Usage')
    ax5.grid(True, alpha=0.3)
    
    # Best scores bar plot
    ax6 = plt.subplot(2, 3, 6)
    metrics = ['Train Acc', 'Val Acc', 'Val F1', 'Val Precision', 'Val Recall']
    values = [
        max(history['train_acc']),
        max(history['val_acc']),
        max(history['val_f1']),
        max(history['val_precision']),
        max(history['val_recall'])
    ]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    bars = ax6.bar(metrics, values, color=colors)
    ax6.set_ylabel('Score')
    ax6.set_title('Best Scores Achieved')
    ax6.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.suptitle(f'{model_name} GPU Training Results', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{model_name}_gpu_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()