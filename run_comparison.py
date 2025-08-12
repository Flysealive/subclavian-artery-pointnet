"""
Comprehensive Comparison: Voxel-Only vs Voxel+Measurements
===========================================================
This script trains both models and provides detailed comparison
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import models and datasets
from voxel_cnn_model import VoxelCNN
from voxel_model_with_measurements import HybridVoxelMeasurementNet
from voxel_dataset import get_voxel_dataloaders
from voxel_dataset_with_measurements import get_measurement_dataloaders
from stl_to_voxel import convert_dataset_to_voxels

class ModelComparison:
    def __init__(self, device='cuda', epochs=50, batch_size=16):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_amp = torch.cuda.is_available()
        
        if self.use_amp:
            self.scaler = GradScaler()
        
        self.results = {
            'voxel_only': {},
            'voxel_with_measurements': {}
        }
        
        print(f"[INFO] Device: {self.device}")
        if torch.cuda.is_available():
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True
    
    def train_model(self, model, train_loader, val_loader, model_name, use_measurements=False):
        """Train a single model and return results"""
        print(f"\n{'='*70}")
        print(f"TRAINING: {model_name}")
        print(f"{'='*70}")
        
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_precision': [], 'val_recall': [], 
            'val_f1': [], 'val_auc': [],
            'epoch_times': []
        }
        
        best_val_acc = 0
        best_val_auc = 0
        best_epoch = 0
        patience = 15
        patience_counter = 0
        
        total_start = time.time()
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Training phase
            model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", ncols=100)
            for batch in pbar:
                if use_measurements:
                    voxels, measurements, labels = batch
                    voxels = voxels.to(self.device)
                    measurements = measurements.to(self.device)
                    labels = labels.to(self.device)
                else:
                    voxels, labels = batch
                    voxels = voxels.to(self.device)
                    labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                if self.use_amp:
                    with autocast():
                        if use_measurements:
                            outputs = model(voxels, measurements)
                        else:
                            outputs = model(voxels)
                        loss = criterion(outputs, labels)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    if use_measurements:
                        outputs = model(voxels, measurements)
                    else:
                        outputs = model(voxels)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Calculate training metrics
            train_acc = accuracy_score(train_labels, train_preds)
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            val_probs = []
            
            with torch.no_grad():
                for batch in val_loader:
                    if use_measurements:
                        voxels, measurements, labels = batch
                        voxels = voxels.to(self.device)
                        measurements = measurements.to(self.device)
                        labels = labels.to(self.device)
                        outputs = model(voxels, measurements)
                    else:
                        voxels, labels = batch
                        voxels = voxels.to(self.device)
                        labels = labels.to(self.device)
                        outputs = model(voxels)
                    
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
                    val_probs.extend(probs[:, 1].cpu().numpy())
            
            # Calculate validation metrics
            avg_val_loss = val_loss / len(val_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            val_precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
            val_recall = recall_score(val_labels, val_preds, average='weighted', zero_division=0)
            val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
            
            try:
                val_auc = roc_auc_score(val_labels, val_probs)
            except:
                val_auc = 0.0
            
            epoch_time = time.time() - epoch_start
            
            # Store history
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            history['val_precision'].append(val_precision)
            history['val_recall'].append(val_recall)
            history['val_f1'].append(val_f1)
            history['val_auc'].append(val_auc)
            history['epoch_times'].append(epoch_time)
            
            # Print epoch results
            print(f"Epoch {epoch+1}: Time={epoch_time:.1f}s")
            print(f"  Train: Loss={avg_train_loss:.4f}, Acc={train_acc:.4f}")
            print(f"  Val:   Loss={avg_val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}")
            
            # Check for best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_auc = val_auc
                best_epoch = epoch + 1
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'val_auc': val_auc
                }, f'best_{model_name}.pth')
                
                print(f"  [SAVE] New best model! Acc={val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n[STOP] Early stopping at epoch {epoch+1}")
                break
        
        total_time = time.time() - total_start
        
        # Final evaluation on best model
        checkpoint = torch.load(f'best_{model_name}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Get final predictions for confusion matrix
        model.eval()
        final_preds = []
        final_labels = []
        final_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                if use_measurements:
                    voxels, measurements, labels = batch
                    voxels = voxels.to(self.device)
                    measurements = measurements.to(self.device)
                    labels = labels.to(self.device)
                    outputs = model(voxels, measurements)
                else:
                    voxels, labels = batch
                    voxels = voxels.to(self.device)
                    labels = labels.to(self.device)
                    outputs = model(voxels)
                
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                final_preds.extend(predicted.cpu().numpy())
                final_labels.extend(labels.cpu().numpy())
                final_probs.extend(probs[:, 1].cpu().numpy())
        
        # Calculate final metrics
        final_cm = confusion_matrix(final_labels, final_preds)
        
        results = {
            'history': history,
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
            'best_val_auc': best_val_auc,
            'total_time': total_time,
            'confusion_matrix': final_cm.tolist(),
            'final_predictions': final_preds,
            'final_labels': final_labels,
            'final_probabilities': final_probs
        }
        
        print(f"\n[COMPLETE] Training time: {total_time/60:.1f} minutes")
        print(f"[COMPLETE] Best accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
        
        return results
    
    def run_comparison(self):
        """Run both models and compare"""
        print("\n" + "="*70)
        print("STARTING COMPARISON EXPERIMENT")
        print("="*70)
        
        # Step 1: Convert STL to voxels if needed
        if not os.path.exists('voxel_arrays'):
            print("\n[STEP 1] Converting STL files to voxels...")
            convert_dataset_to_voxels(voxel_size=64)
        else:
            print("\n[STEP 1] Voxel arrays already exist")
        
        # Step 2: Train model WITHOUT measurements
        print("\n[STEP 2] Training Voxel-Only Model")
        
        # Load data without measurements
        train_loader_basic, val_loader_basic = get_voxel_dataloaders(
            batch_size=self.batch_size,
            balanced=True,
            num_workers=4 if torch.cuda.is_available() else 0
        )
        
        # Create and train basic model
        model_basic = VoxelCNN(num_classes=2, voxel_size=64)
        total_params = sum(p.numel() for p in model_basic.parameters())
        print(f"Model parameters: {total_params:,}")
        
        self.results['voxel_only'] = self.train_model(
            model_basic, 
            train_loader_basic, 
            val_loader_basic,
            'voxel_only',
            use_measurements=False
        )
        
        # Step 3: Train model WITH measurements
        print("\n[STEP 3] Training Voxel+Measurements Model")
        
        # Load data with measurements
        train_loader_enhanced, val_loader_enhanced = get_measurement_dataloaders(
            batch_size=self.batch_size,
            balanced=True,
            num_workers=4 if torch.cuda.is_available() else 0
        )
        
        # Create and train enhanced model
        model_enhanced = HybridVoxelMeasurementNet(num_classes=2, voxel_size=64, num_measurements=3)
        total_params = sum(p.numel() for p in model_enhanced.parameters())
        print(f"Model parameters: {total_params:,}")
        
        self.results['voxel_with_measurements'] = self.train_model(
            model_enhanced,
            train_loader_enhanced,
            val_loader_enhanced,
            'voxel_with_measurements',
            use_measurements=True
        )
        
        # Step 4: Generate comparison report
        self.generate_comparison_report()
        
        return self.results
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison plots and report"""
        print("\n[STEP 4] Generating Comparison Report")
        
        # Create comparison plots
        fig = plt.figure(figsize=(20, 12))
        
        # Plot 1: Accuracy comparison over epochs
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(self.results['voxel_only']['history']['val_acc'], 
                label='Voxel Only', linewidth=2, color='blue')
        ax1.plot(self.results['voxel_with_measurements']['history']['val_acc'], 
                label='Voxel + Measurements', linewidth=2, color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Validation Accuracy')
        ax1.set_title('Accuracy Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Loss comparison
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(self.results['voxel_only']['history']['val_loss'], 
                label='Voxel Only', linewidth=2, color='blue')
        ax2.plot(self.results['voxel_with_measurements']['history']['val_loss'], 
                label='Voxel + Measurements', linewidth=2, color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Loss Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: F1 Score comparison
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(self.results['voxel_only']['history']['val_f1'], 
                label='Voxel Only', linewidth=2, color='blue')
        ax3.plot(self.results['voxel_with_measurements']['history']['val_f1'], 
                label='Voxel + Measurements', linewidth=2, color='red')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('F1 Score Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: AUC comparison
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(self.results['voxel_only']['history']['val_auc'], 
                label='Voxel Only', linewidth=2, color='blue')
        ax4.plot(self.results['voxel_with_measurements']['history']['val_auc'], 
                label='Voxel + Measurements', linewidth=2, color='red')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('ROC-AUC')
        ax4.set_title('ROC-AUC Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Confusion Matrix - Voxel Only
        ax5 = plt.subplot(3, 3, 5)
        cm1 = np.array(self.results['voxel_only']['confusion_matrix'])
        sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=ax5,
                   xticklabels=['Class 0', 'Class 1'],
                   yticklabels=['Class 0', 'Class 1'])
        ax5.set_title('Confusion Matrix: Voxel Only')
        ax5.set_ylabel('True Label')
        ax5.set_xlabel('Predicted Label')
        
        # Plot 6: Confusion Matrix - Voxel + Measurements
        ax6 = plt.subplot(3, 3, 6)
        cm2 = np.array(self.results['voxel_with_measurements']['confusion_matrix'])
        sns.heatmap(cm2, annot=True, fmt='d', cmap='Reds', ax=ax6,
                   xticklabels=['Class 0', 'Class 1'],
                   yticklabels=['Class 0', 'Class 1'])
        ax6.set_title('Confusion Matrix: Voxel + Measurements')
        ax6.set_ylabel('True Label')
        ax6.set_xlabel('Predicted Label')
        
        # Plot 7: Metrics comparison bar chart
        ax7 = plt.subplot(3, 3, 7)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        voxel_only_scores = [
            self.results['voxel_only']['best_val_acc'],
            max(self.results['voxel_only']['history']['val_precision']),
            max(self.results['voxel_only']['history']['val_recall']),
            max(self.results['voxel_only']['history']['val_f1']),
            self.results['voxel_only']['best_val_auc']
        ]
        voxel_meas_scores = [
            self.results['voxel_with_measurements']['best_val_acc'],
            max(self.results['voxel_with_measurements']['history']['val_precision']),
            max(self.results['voxel_with_measurements']['history']['val_recall']),
            max(self.results['voxel_with_measurements']['history']['val_f1']),
            self.results['voxel_with_measurements']['best_val_auc']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax7.bar(x - width/2, voxel_only_scores, width, label='Voxel Only', color='blue', alpha=0.7)
        bars2 = ax7.bar(x + width/2, voxel_meas_scores, width, label='Voxel + Measurements', color='red', alpha=0.7)
        
        ax7.set_xlabel('Metrics')
        ax7.set_ylabel('Score')
        ax7.set_title('Best Metrics Comparison')
        ax7.set_xticks(x)
        ax7.set_xticklabels(metrics)
        ax7.legend()
        ax7.set_ylim([0, 1.1])
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax7.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        # Plot 8: Training time comparison
        ax8 = plt.subplot(3, 3, 8)
        times = [
            self.results['voxel_only']['total_time'] / 60,
            self.results['voxel_with_measurements']['total_time'] / 60
        ]
        models = ['Voxel Only', 'Voxel +\nMeasurements']
        colors = ['blue', 'red']
        bars = ax8.bar(models, times, color=colors, alpha=0.7)
        ax8.set_ylabel('Training Time (minutes)')
        ax8.set_title('Training Time Comparison')
        
        for bar, time in zip(bars, times):
            ax8.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{time:.1f} min', ha='center', va='bottom')
        
        # Plot 9: Improvement summary
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Calculate improvements
        acc_improvement = ((self.results['voxel_with_measurements']['best_val_acc'] - 
                          self.results['voxel_only']['best_val_acc']) / 
                         self.results['voxel_only']['best_val_acc'] * 100)
        
        auc_improvement = ((self.results['voxel_with_measurements']['best_val_auc'] - 
                          self.results['voxel_only']['best_val_auc']) / 
                         max(self.results['voxel_only']['best_val_auc'], 0.01) * 100)
        
        summary_text = f"""
COMPARISON SUMMARY
==================

Voxel Only Model:
  • Best Accuracy: {self.results['voxel_only']['best_val_acc']:.4f}
  • Best AUC: {self.results['voxel_only']['best_val_auc']:.4f}
  • Best Epoch: {self.results['voxel_only']['best_epoch']}
  • Training Time: {self.results['voxel_only']['total_time']/60:.1f} min

Voxel + Measurements Model:
  • Best Accuracy: {self.results['voxel_with_measurements']['best_val_acc']:.4f}
  • Best AUC: {self.results['voxel_with_measurements']['best_val_auc']:.4f}
  • Best Epoch: {self.results['voxel_with_measurements']['best_epoch']}
  • Training Time: {self.results['voxel_with_measurements']['total_time']/60:.1f} min

Improvements with Measurements:
  • Accuracy: {acc_improvement:+.1f}%
  • AUC: {auc_improvement:+.1f}%
"""
        
        ax9.text(0.5, 0.5, summary_text, transform=ax9.transAxes,
                fontsize=11, verticalalignment='center',
                horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Voxel CNN Model Comparison: With vs Without Anatomical Measurements', 
                    fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig('model_comparison_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Save detailed results to JSON
        comparison_results = {
            'voxel_only': {
                'best_accuracy': float(self.results['voxel_only']['best_val_acc']),
                'best_auc': float(self.results['voxel_only']['best_val_auc']),
                'best_epoch': int(self.results['voxel_only']['best_epoch']),
                'training_time_minutes': float(self.results['voxel_only']['total_time'] / 60),
                'history': {k: [float(v) for v in vals] for k, vals in self.results['voxel_only']['history'].items()}
            },
            'voxel_with_measurements': {
                'best_accuracy': float(self.results['voxel_with_measurements']['best_val_acc']),
                'best_auc': float(self.results['voxel_with_measurements']['best_val_auc']),
                'best_epoch': int(self.results['voxel_with_measurements']['best_epoch']),
                'training_time_minutes': float(self.results['voxel_with_measurements']['total_time'] / 60),
                'history': {k: [float(v) for v in vals] for k, vals in self.results['voxel_with_measurements']['history'].items()}
            },
            'improvements': {
                'accuracy_improvement_percent': float(acc_improvement),
                'auc_improvement_percent': float(auc_improvement)
            }
        }
        
        with open('comparison_results.json', 'w') as f:
            json.dump(comparison_results, f, indent=4)
        
        print("\n[SAVED] Results saved to:")
        print("  • model_comparison_results.png")
        print("  • comparison_results.json")
        print("  • best_voxel_only.pth")
        print("  • best_voxel_with_measurements.pth")

def main():
    print("\n" + "="*70)
    print("VOXEL CNN COMPARISON EXPERIMENT")
    print("Comparing: Voxel-Only vs Voxel+Anatomical Measurements")
    print("="*70)
    
    # Run comparison
    comparison = ModelComparison(epochs=50, batch_size=16)
    results = comparison.run_comparison()
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()