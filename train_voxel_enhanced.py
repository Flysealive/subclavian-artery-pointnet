"""
Enhanced Voxel Training with Optional Anatomical Measurements
==============================================================
Train with or without anatomical measurements for improved accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
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

# Import models
from voxel_cnn_model import VoxelCNN, VoxelResNet
from voxel_model_with_measurements import (
    VoxelCNNWithMeasurements, 
    AttentionFusionVoxelCNN, 
    HybridVoxelMeasurementNet
)

# Import datasets
from voxel_dataset import get_voxel_dataloaders
from voxel_dataset_with_measurements import get_measurement_dataloaders

class EnhancedVoxelTrainer:
    def __init__(self, model, device, model_name="voxel_model", use_measurements=False):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.use_measurements = use_measurements
        self.use_amp = torch.cuda.is_available()
        
        if self.use_amp:
            self.scaler = GradScaler()
            print("[OK] Using Automatic Mixed Precision (AMP)")
        
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_precision': [], 'val_recall': [], 
            'val_f1': [], 'val_auc': [],
            'learning_rates': []
        }
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            print("[OK] CUDNN benchmarking enabled")
    
    def train_epoch(self, train_loader, criterion, optimizer, scheduler=None):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(train_loader, desc="Training", ncols=100)
        
        for batch_idx, batch in enumerate(pbar):
            if self.use_measurements:
                voxels, measurements, labels = batch
                voxels = voxels.to(self.device, non_blocking=True)
                measurements = measurements.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
            else:
                voxels, labels = batch
                voxels = voxels.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with AMP
            if self.use_amp:
                with autocast():
                    if self.use_measurements:
                        outputs = self.model(voxels, measurements)
                    else:
                        outputs = self.model(voxels)
                    loss = criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                if self.use_measurements:
                    outputs = self.model(voxels, measurements)
                else:
                    outputs = self.model(voxels)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            
            if scheduler and isinstance(scheduler, OneCycleLR):
                scheduler.step()
            
            # Collect predictions
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())  # Probability of class 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc, all_probs, all_labels
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", ncols=100):
                if self.use_measurements:
                    voxels, measurements, labels = batch
                    voxels = voxels.to(self.device, non_blocking=True)
                    measurements = measurements.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                else:
                    voxels, labels = batch
                    voxels = voxels.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with autocast():
                        if self.use_measurements:
                            outputs = self.model(voxels, measurements)
                        else:
                            outputs = self.model(voxels)
                        loss = criterion(outputs, labels)
                else:
                    if self.use_measurements:
                        outputs = self.model(voxels, measurements)
                    else:
                        outputs = self.model(voxels)
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].detach().cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        metrics = self.calculate_metrics(all_labels, all_preds, all_probs)
        
        return epoch_loss, metrics, all_labels, all_preds
    
    def calculate_metrics(self, y_true, y_pred, y_prob=None):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        if y_prob is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['auc'] = 0.0
        else:
            metrics['auc'] = 0.0
        
        return metrics
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, patience=20):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Learning rate scheduler
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=lr*10,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        best_val_acc = 0
        best_val_auc = 0
        best_epoch = 0
        patience_counter = 0
        
        print("\n" + "="*60)
        print(f"Training {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Using measurements: {self.use_measurements}")
        print(f"Epochs: {epochs} | LR: {lr}")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 40)
            
            # Training
            train_loss, train_acc, train_probs, train_labels = self.train_epoch(
                train_loader, criterion, optimizer, scheduler
            )
            
            # Validation
            val_loss, val_metrics, val_labels, val_preds = self.validate(
                val_loader, criterion
            )
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Print metrics
            epoch_time = time.time() - epoch_start
            print(f"Time: {epoch_time:.1f}s")
            print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val   - Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f}")
            print(f"Val   - F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['auc']:.4f}")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_val_auc = val_metrics['auc']
                best_epoch = epoch + 1
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'best_val_auc': best_val_auc,
                    'history': self.history,
                    'use_measurements': self.use_measurements
                }
                
                torch.save(checkpoint, f'best_{self.model_name}.pth')
                print(f"[SAVE] New best model! Acc: {best_val_acc:.4f}, AUC: {best_val_auc:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n[STOP] Early stopping at epoch {epoch+1}")
                break
        
        # Training complete
        total_time = (time.time() - start_time) / 60
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print(f"Total Time: {total_time:.1f} minutes")
        print(f"Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
        print(f"Best Val AUC: {best_val_auc:.4f}")
        print("="*60)
        
        # Generate confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        self.plot_results(cm)
        
        return self.history
    
    def plot_results(self, cm):
        # Create comprehensive plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Class 0', 'Class 1'],
                   yticklabels=['Class 0', 'Class 1'],
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # Loss curves
        axes[0, 1].plot(self.history['train_loss'], label='Train', linewidth=2)
        axes[0, 1].plot(self.history['val_loss'], label='Val', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Loss Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 2].plot(self.history['train_acc'], label='Train', linewidth=2)
        axes[0, 2].plot(self.history['val_acc'], label='Val', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].set_title('Accuracy Curves')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Metrics comparison
        axes[1, 0].plot(self.history['val_precision'], label='Precision', linewidth=2)
        axes[1, 0].plot(self.history['val_recall'], label='Recall', linewidth=2)
        axes[1, 0].plot(self.history['val_f1'], label='F1', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Validation Metrics')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # AUC curve
        axes[1, 1].plot(self.history['val_auc'], linewidth=2, color='green')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].set_title('ROC-AUC Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Final scores bar plot
        final_scores = {
            'Accuracy': max(self.history['val_acc']),
            'Precision': max(self.history['val_precision']),
            'Recall': max(self.history['val_recall']),
            'F1': max(self.history['val_f1']),
            'AUC': max(self.history['val_auc'])
        }
        
        bars = axes[1, 2].bar(final_scores.keys(), final_scores.values(), 
                             color=['blue', 'green', 'orange', 'red', 'purple'])
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_title('Best Scores Achieved')
        axes[1, 2].set_ylim([0, 1.1])
        
        for bar, value in zip(bars, final_scores.values()):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                          f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle(f'{self.model_name} Results {"(with measurements)" if self.use_measurements else "(voxel only)"}', 
                    fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{self.model_name}_results.png', dpi=150, bbox_inches='tight')
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Enhanced Voxel Training')
    parser.add_argument('--use_measurements', action='store_true', 
                       help='Use anatomical measurements')
    parser.add_argument('--model', type=str, default='hybrid',
                       choices=['cnn', 'resnet', 'combined', 'attention', 'hybrid'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--voxel_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    
    # Convert STL to voxels if needed
    if not os.path.exists('voxel_arrays'):
        print("\n[INFO] Converting STL files to voxels...")
        from stl_to_voxel import convert_dataset_to_voxels
        convert_dataset_to_voxels(voxel_size=args.voxel_size)
    
    # Load appropriate dataset
    if args.use_measurements:
        print("\n[INFO] Loading dataset WITH anatomical measurements")
        train_loader, test_loader = get_measurement_dataloaders(
            batch_size=args.batch_size,
            balanced=True,
            num_workers=args.num_workers
        )
        
        # Select model for measurements
        if args.model == 'combined':
            model = VoxelCNNWithMeasurements(num_classes=2, voxel_size=args.voxel_size)
            model_name = "voxel_cnn_measurements"
        elif args.model == 'attention':
            model = AttentionFusionVoxelCNN(num_classes=2, voxel_size=args.voxel_size)
            model_name = "attention_fusion_voxel"
        else:  # hybrid
            model = HybridVoxelMeasurementNet(num_classes=2, voxel_size=args.voxel_size)
            model_name = "hybrid_voxel_measurement"
    else:
        print("\n[INFO] Loading dataset WITHOUT measurements (voxel only)")
        train_loader, test_loader = get_voxel_dataloaders(
            batch_size=args.batch_size,
            balanced=True,
            num_workers=args.num_workers
        )
        
        # Select model for voxel only
        if args.model == 'resnet':
            model = VoxelResNet(num_classes=2, voxel_size=args.voxel_size)
            model_name = "voxel_resnet"
        else:
            model = VoxelCNN(num_classes=2, voxel_size=args.voxel_size)
            model_name = "voxel_cnn"
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[INFO] Model: {model_name}")
    print(f"[INFO] Parameters: {total_params:,}")
    
    # Train
    trainer = EnhancedVoxelTrainer(
        model, device, model_name, 
        use_measurements=args.use_measurements
    )
    
    history = trainer.train(
        train_loader, test_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience
    )
    
    # Save history
    with open(f'{model_name}_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"\n[OK] Training complete! Results saved.")
    print(f"[OK] Model: best_{model_name}.pth")
    print(f"[OK] History: {model_name}_history.json")
    print(f"[OK] Plots: {model_name}_results.png")

if __name__ == "__main__":
    main()