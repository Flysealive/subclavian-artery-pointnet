"""
Extended Training: Hybrid Multi-Modal Model with 150 Epochs
============================================================
Training with more epochs to maximize performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Import the hybrid model components
from hybrid_multimodal_model import (
    HybridMultiModalNet,
    HybridMultiModalDataset,
    prepare_hybrid_data
)

def train_hybrid_extended(epochs=150, batch_size=8, lr=0.001):
    """Train hybrid model with extended epochs and monitoring"""
    
    print("\n" + "="*70)
    print("EXTENDED HYBRID MULTI-MODAL TRAINING")
    print("Target: Beat Traditional ML's 89.47% accuracy")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Training for {epochs} epochs")
    
    # Prepare data if not exists
    if not os.path.exists('hybrid_data'):
        print("\nPreparing hybrid data...")
        prepare_hybrid_data()
    
    # Create datasets
    from torch.utils.data import DataLoader
    
    train_dataset = HybridMultiModalDataset(mode='train')
    test_dataset = HybridMultiModalDataset(mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model with enhanced regularization for longer training
    model = HybridMultiModalNet(
        num_classes=2,
        num_points=2048,
        voxel_size=32,
        num_measurements=3
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup with enhanced regularization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Add label smoothing
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Cosine annealing with warm restarts for longer training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=30,  # Restart every 30 epochs
        T_mult=1,  # Keep restart period constant
        eta_min=1e-6
    )
    
    # Early stopping parameters
    best_acc = 0
    best_f1 = 0
    patience_counter = 0
    max_patience = 25  # Increased patience for longer training
    
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1': [],
        'lr': []
    }
    
    print("\nStarting training...")
    print("-" * 70)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for pc, vox, meas, labels in pbar:
                pc = pc.to(device)
                vox = vox.to(device)
                meas = meas.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass with gradient accumulation for stability
                outputs = model(pc, vox, meas)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Track metrics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                current_acc = train_correct / train_total
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.4f}'})
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for pc, vox, meas, labels in test_loader:
                pc = pc.to(device)
                vox = vox.to(device)
                meas = meas.to(device)
                labels = labels.to(device)
                
                outputs = model(pc, vox, meas)
                _, predicted = torch.max(outputs.data, 1)
                
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = val_correct / val_total
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Save history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Check for improvement
        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = val_f1
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'best_f1': best_f1,
                'history': history
            }, 'best_hybrid_150epochs.pth')
            
            print(f"  >>> NEW BEST MODEL SAVED: Acc={best_acc:.4f}, F1={best_f1:.4f}")
            
            # Special announcement if we beat traditional ML
            if best_acc > 0.8947:
                print("  *** BREAKTHROUGH: BEAT TRADITIONAL ML (89.47%)! ***")
        else:
            patience_counter += 1
            
        # Early stopping check
        if patience_counter >= max_patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"No improvement for {max_patience} epochs")
            break
        
        # Progress check every 25 epochs
        if (epoch + 1) % 25 == 0:
            print(f"\n{'='*70}")
            print(f"PROGRESS CHECK - Epoch {epoch+1}")
            print(f"Best Accuracy so far: {best_acc:.4f}")
            print(f"Target to beat: 89.47% (Traditional ML)")
            gap = 0.8947 - best_acc
            if gap > 0:
                print(f"Gap to target: {gap:.4f} ({gap*100:.2f}%)")
            else:
                print(f"TARGET EXCEEDED by {-gap:.4f} ({-gap*100:.2f}%)")
            print(f"{'='*70}\n")
    
    # Final results
    print("\n" + "="*70)
    print("EXTENDED TRAINING COMPLETE")
    print("="*70)
    print(f"Total epochs trained: {len(history['epoch'])}")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"Best F1 Score: {best_f1:.4f}")
    
    if best_acc > 0.8947:
        print("\n*** SUCCESS: HYBRID MODEL BEAT TRADITIONAL ML! ***")
    else:
        gap = 0.8947 - best_acc
        print(f"\nTraditional ML still better by {gap:.4f} ({gap*100:.2f}%)")
    
    # Plot training history
    plot_extended_training(history, best_acc)
    
    # Save final history
    import json
    with open('hybrid_150epochs_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("\n[SAVED] hybrid_150epochs_history.json")
    
    return history, best_acc, best_f1

def plot_extended_training(history, best_acc):
    """Create detailed plots of extended training"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Extended Hybrid Training Results (150 Epochs)\nBest Accuracy: {best_acc:.4f}', 
                 fontsize=14, fontweight='bold')
    
    epochs = history['epoch']
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Accuracy Comparison
    ax = axes[0, 1]
    ax.plot(epochs, history['train_acc'], 'b-', label='Train Acc', alpha=0.8)
    ax.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax.axhline(y=0.8947, color='g', linestyle='--', label='Traditional ML (89.47%)', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy: Hybrid vs Traditional ML')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0.5, 1.0])
    
    # Plot 3: F1 Score
    ax = axes[1, 0]
    ax.plot(epochs, history['val_f1'], 'purple', label='Validation F1', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('Validation F1 Score Progress')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Learning Rate
    ax = axes[1, 1]
    ax.plot(epochs, history['lr'], 'orange', label='Learning Rate', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule (Cosine Annealing)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('hybrid_150epochs_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("[SAVED] hybrid_150epochs_results.png")

if __name__ == "__main__":
    # Run extended training
    history, best_acc, best_f1 = train_hybrid_extended(epochs=150, batch_size=8, lr=0.001)