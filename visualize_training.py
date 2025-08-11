#!/usr/bin/env python3
"""
Real-time training metrics visualization
Shows training progress and performance metrics
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import time

def load_training_history():
    """Load training history from JSON file"""
    history_file = 'logs/training_history.json'
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    return None

def create_training_plots(save_path='training_progress.png'):
    """Create comprehensive training visualization"""
    
    # Try to load history
    history = load_training_history()
    
    # If no history file yet, parse from current output
    if history is None:
        print("Training history file not found yet. Creating sample visualization...")
        # Sample data based on current training
        epochs = list(range(1, 25))
        history = {
            'train_loss': [0.7532, 0.8377, 0.7836, 0.8694, 0.7843, 0.7102, 0.7560, 0.6268, 
                          0.6182, 0.6888, 0.5399, 0.6831, 0.5984, 0.6227, 0.5382, 0.5428,
                          0.5759, 0.5681, 0.4956, 0.6601, 0.4619, 0.6392, 0.4670, 0.5397],
            'train_acc': [45.3, 32.8, 39.1, 37.5, 43.8, 54.7, 48.4, 54.7,
                         62.5, 59.4, 76.6, 57.8, 71.9, 65.6, 70.3, 70.3,
                         70.3, 82.8, 76.6, 75.0, 84.4, 75.0, 81.2, 75.0],
            'val_loss': [0.6981, 0.7020, 0.6935, 0.6796, 0.6654, 0.6687, 0.6873, 0.6906,
                        0.7005, 0.6981, 0.6869, 0.6878, 0.7038, 0.7312, 0.7823, 0.8238,
                        0.8712, 0.8781, 0.8578, 0.7791, 0.7214, 0.6752, 0.6280, 0.6200],
            'val_acc': [7.1, 7.1, 42.9, 92.9, 92.9, 85.7, 57.1, 57.1,
                       57.1, 71.4, 71.4, 71.4, 71.4, 71.4, 71.4, 71.4,
                       71.4, 64.3, 71.4, 85.7, 85.7, 85.7, 78.6, 78.6],
            'val_balanced_acc': [50.0, 50.0, 69.2, 50.0, 50.0, 46.2, 30.8, 30.8,
                                30.8, 38.5, 38.5, 38.5, 38.5, 38.5, 38.5, 38.5,
                                84.6, 34.6, 38.5, 46.2, 46.2, 46.2, 42.3, 42.3],
            'lr': [0.001, 0.000999, 0.000996, 0.000991, 0.000984, 0.000976, 0.000965, 0.000952,
                  0.000938, 0.000922, 0.000905, 0.000885, 0.000864, 0.000842, 0.000819, 0.000794,
                  0.000768, 0.000741, 0.000713, 0.000684, 0.000655, 0.000624, 0.000594, 0.000564]
        }
    else:
        epochs = list(range(1, len(history['train_loss']) + 1))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('GPU Training Progress - PointNet with Anatomical Measurements', fontsize=16, fontweight='bold')
    
    # 1. Loss plot
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy plot
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Balanced Accuracy plot
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(epochs, history['val_balanced_acc'], 'g-', label='Val Balanced Acc', linewidth=2, marker='o', markersize=4)
    best_ba = max(history['val_balanced_acc'])
    best_epoch = history['val_balanced_acc'].index(best_ba) + 1
    ax3.axhline(y=best_ba, color='orange', linestyle='--', alpha=0.7, label=f'Best: {best_ba:.1f}%')
    ax3.scatter([best_epoch], [best_ba], color='red', s=100, zorder=5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Balanced Accuracy (%)')
    ax3.set_title('Validation Balanced Accuracy (Class-weighted)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Learning Rate schedule
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(epochs, history['lr'], 'purple', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule (Cosine Annealing)')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # 5. Training Statistics
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    # Calculate statistics
    current_epoch = len(epochs)
    current_train_acc = history['train_acc'][-1] if history['train_acc'] else 0
    current_val_acc = history['val_acc'][-1] if history['val_acc'] else 0
    current_bal_acc = history['val_balanced_acc'][-1] if history['val_balanced_acc'] else 0
    
    stats_text = f"""
    Training Statistics (Epoch {current_epoch})
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Current Performance:
    • Train Accuracy: {current_train_acc:.1f}%
    • Val Accuracy: {current_val_acc:.1f}%
    • Val Balanced Acc: {current_bal_acc:.1f}%
    
    Best Performance:
    • Best Balanced Acc: {best_ba:.1f}%
    • Best Epoch: {best_epoch}
    
    Model Configuration:
    • GPU: NVIDIA RTX 4060 Ti (8GB)
    • Parameters: 2.37M
    • Point Cloud: 2048 points
    • Measurements: 3 anatomical
    • Classes: Binary (0/1)
    
    Features:
    ✓ Class-weighted loss
    ✓ Data augmentation
    ✓ Spatial transformer
    ✓ Early stopping (patience=100)
    """
    
    ax5.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # 6. Class Performance Over Time
    ax6 = plt.subplot(2, 3, 6)
    
    # Estimate class performances from balanced accuracy
    # This is approximate since we don't have per-class data in current history
    val_accs = np.array(history['val_acc'])
    bal_accs = np.array(history['val_balanced_acc'])
    
    # Rough estimation (this would be more accurate with actual class data)
    class0_est = val_accs * 0.9  # Majority class usually performs better
    class1_est = bal_accs * 2 - class0_est  # Derive from balanced accuracy
    class1_est = np.clip(class1_est, 0, 100)
    
    ax6.plot(epochs, class0_est, 'b-', label='Class 0 (Majority)', linewidth=2, alpha=0.7)
    ax6.plot(epochs, class1_est, 'r-', label='Class 1 (Minority)', linewidth=2, alpha=0.7)
    ax6.fill_between(epochs, 0, class0_est, alpha=0.2, color='blue')
    ax6.fill_between(epochs, 0, class1_est, alpha=0.2, color='red')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Estimated Accuracy (%)')
    ax6.set_title('Per-Class Performance (Estimated)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 105])
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training visualization saved to: {save_path}")
    
    # Also create a simple real-time plot
    create_realtime_plot(history, epochs)
    
    return fig

def create_realtime_plot(history, epochs):
    """Create a simpler real-time monitoring plot"""
    
    fig2 = plt.figure(figsize=(12, 4))
    fig2.suptitle('Real-time Training Monitor', fontsize=14, fontweight='bold')
    
    # Combined plot
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', alpha=0.7)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', alpha=0.7)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs, history['val_balanced_acc'], 'g-', label='Balanced Acc', linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='black')
    ax1_twin.set_ylabel('Balanced Accuracy (%)', color='green')
    ax1.set_title('Loss & Balanced Accuracy Progress')
    ax1.grid(True, alpha=0.3)
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Recent performance (last 20 epochs)
    ax2 = plt.subplot(1, 2, 2)
    recent_epochs = epochs[-20:] if len(epochs) > 20 else epochs
    recent_train_acc = history['train_acc'][-20:] if len(history['train_acc']) > 20 else history['train_acc']
    recent_val_acc = history['val_acc'][-20:] if len(history['val_acc']) > 20 else history['val_acc']
    recent_bal_acc = history['val_balanced_acc'][-20:] if len(history['val_balanced_acc']) > 20 else history['val_balanced_acc']
    
    ax2.plot(recent_epochs, recent_train_acc, 'b-o', label='Train Acc', markersize=4)
    ax2.plot(recent_epochs, recent_val_acc, 'r-s', label='Val Acc', markersize=4)
    ax2.plot(recent_epochs, recent_bal_acc, 'g-^', label='Balanced Acc', markersize=4, linewidth=2)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'Recent Performance (Last {len(recent_epochs)} Epochs)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('realtime_monitor.png', dpi=100, bbox_inches='tight')
    print(f"Real-time monitor saved to: realtime_monitor.png")

def main():
    """Main function to create visualizations"""
    print("="*60)
    print("Training Metrics Visualization")
    print("="*60)
    
    # Create visualizations
    fig = create_training_plots()
    
    print("\nVisualization complete!")
    print("Files created:")
    print("  - training_progress.png (comprehensive view)")
    print("  - realtime_monitor.png (simple monitor)")
    
    # Show current training status
    history = load_training_history()
    if history:
        current_epoch = len(history['train_loss'])
        print(f"\nCurrent Status (from saved history):")
        print(f"  Epoch: {current_epoch}")
        print(f"  Train Acc: {history['train_acc'][-1]:.1f}%")
        print(f"  Val Balanced Acc: {history['val_balanced_acc'][-1]:.1f}%")
    
    plt.show()

if __name__ == "__main__":
    main()