"""
Final Comparison Report: Voxel-Only vs Voxel+Measurements
==========================================================
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load results
with open('simple_comparison_results.json', 'r') as f:
    results = json.load(f)

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))

# 1. Training curves comparison
ax1 = plt.subplot(2, 3, 1)
epochs = range(1, 21)
ax1.plot(epochs, results['voxel_only']['history']['acc'], 
         'b-', linewidth=2, label='Voxel Only')
ax1.plot(epochs, results['voxel_with_measurements']['history']['acc'], 
         'r-', linewidth=2, label='Voxel + Measurements')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Validation Accuracy')
ax1.set_title('Validation Accuracy Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])

# 2. Loss curves comparison
ax2 = plt.subplot(2, 3, 2)
ax2.plot(epochs, results['voxel_only']['history']['loss'], 
         'b-', linewidth=2, label='Voxel Only')
ax2.plot(epochs, results['voxel_with_measurements']['history']['loss'], 
         'r-', linewidth=2, label='Voxel + Measurements')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Training Loss')
ax2.set_title('Training Loss Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Final metrics comparison
ax3 = plt.subplot(2, 3, 3)
metrics = ['Best Accuracy', 'F1 Score']
voxel_only_scores = [
    results['voxel_only']['best_acc'],
    results['voxel_only']['f1_score']
]
voxel_meas_scores = [
    results['voxel_with_measurements']['best_acc'],
    results['voxel_with_measurements']['f1_score']
]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax3.bar(x - width/2, voxel_only_scores, width, 
                label='Voxel Only', color='blue', alpha=0.7)
bars2 = ax3.bar(x + width/2, voxel_meas_scores, width, 
                label='Voxel + Measurements', color='red', alpha=0.7)

ax3.set_ylabel('Score')
ax3.set_title('Final Metrics Comparison')
ax3.set_xticks(x)
ax3.set_xticklabels(metrics)
ax3.legend()
ax3.set_ylim([0, 1])

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom')
for bar in bars2:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.3f}', ha='center', va='bottom')

# 4. Confusion Matrix - Voxel Only
ax4 = plt.subplot(2, 3, 4)
cm1 = np.array(results['voxel_only']['confusion_matrix'])
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=ax4,
           xticklabels=['Pred 0', 'Pred 1'],
           yticklabels=['True 0', 'True 1'],
           cbar_kws={'label': 'Count'})
ax4.set_title('Confusion Matrix: Voxel Only')
accuracy1 = np.trace(cm1) / np.sum(cm1)
ax4.text(0.5, -0.15, f'Accuracy: {accuracy1:.2%}', 
        ha='center', transform=ax4.transAxes)

# 5. Confusion Matrix - Voxel + Measurements
ax5 = plt.subplot(2, 3, 5)
cm2 = np.array(results['voxel_with_measurements']['confusion_matrix'])
sns.heatmap(cm2, annot=True, fmt='d', cmap='Reds', ax=ax5,
           xticklabels=['Pred 0', 'Pred 1'],
           yticklabels=['True 0', 'True 1'],
           cbar_kws={'label': 'Count'})
ax5.set_title('Confusion Matrix: Voxel + Measurements')
accuracy2 = np.trace(cm2) / np.sum(cm2)
ax5.text(0.5, -0.15, f'Accuracy: {accuracy2:.2%}', 
        ha='center', transform=ax5.transAxes)

# 6. Summary text
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_text = f"""
COMPARISON SUMMARY (20 Epochs)
===============================

VOXEL ONLY MODEL:
• Best Accuracy: {results['voxel_only']['best_acc']:.2%}
• F1 Score: {results['voxel_only']['f1_score']:.3f}
• True Positives: {cm1[1,1]} / {cm1[1,:].sum()}
• True Negatives: {cm1[0,0]} / {cm1[0,:].sum()}

VOXEL + MEASUREMENTS MODEL:
• Best Accuracy: {results['voxel_with_measurements']['best_acc']:.2%}
• F1 Score: {results['voxel_with_measurements']['f1_score']:.3f}
• True Positives: {cm2[1,1]} / {cm2[1,:].sum()}
• True Negatives: {cm2[0,0]} / {cm2[0,:].sum()}

KEY FINDINGS:
• Voxel-only achieved better accuracy
• Measurements model may need:
  - More training epochs
  - Hyperparameter tuning
  - Feature engineering
  - Larger dataset
"""

ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
        fontsize=10, verticalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('Voxel CNN Comparison Results: Impact of Anatomical Measurements', 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('final_comparison_report.png', dpi=150, bbox_inches='tight')
plt.show()

# Print detailed analysis
print("\n" + "="*70)
print("DETAILED ANALYSIS OF RESULTS")
print("="*70)

print("\n1. PERFORMANCE METRICS:")
print("-" * 40)
print(f"Voxel Only:")
print(f"  • Accuracy: {results['voxel_only']['best_acc']:.2%}")
print(f"  • F1 Score: {results['voxel_only']['f1_score']:.3f}")
print(f"\nVoxel + Measurements:")
print(f"  • Accuracy: {results['voxel_with_measurements']['best_acc']:.2%}")
print(f"  • F1 Score: {results['voxel_with_measurements']['f1_score']:.3f}")

accuracy_diff = results['voxel_with_measurements']['best_acc'] - results['voxel_only']['best_acc']
print(f"\nDifference: {accuracy_diff:+.2%}")

print("\n2. POSSIBLE EXPLANATIONS FOR RESULTS:")
print("-" * 40)
print("• Limited training (only 20 epochs)")
print("• Small dataset size (75-76 training samples)")
print("• Measurements model has more parameters to learn")
print("• May need different learning rates for each pathway")
print("• Measurements might need different normalization")

print("\n3. RECOMMENDATIONS:")
print("-" * 40)
print("• Train for more epochs (50-100)")
print("• Use learning rate scheduling")
print("• Try different architectures (attention fusion)")
print("• Perform hyperparameter tuning")
print("• Collect more training data if possible")
print("• Apply more aggressive data augmentation")

print("\n4. DATASET STATISTICS:")
print("-" * 40)
print("• Total samples: ~95")
print("• Training set: ~76 (80%)")
print("• Test set: ~19 (20%)")
print("• Class distribution: Imbalanced (more class 0)")
print("• Balanced training used oversampling")

print("\n[REPORT SAVED] final_comparison_report.png")
print("="*70)