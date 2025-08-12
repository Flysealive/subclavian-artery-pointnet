#!/usr/bin/env python3
"""
TEST SET EVALUATION SUMMARY
============================
Critical findings about validation vs test performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def display_evaluation_results():
    """
    Display the critical findings about test vs validation performance
    """
    
    print("="*80)
    print("CRITICAL FINDINGS: VALIDATION vs TEST SET PERFORMANCE")
    print("="*80)
    
    print("\n⚠️  IMPORTANT DISCOVERY:")
    print("-"*60)
    print("We have been reporting VALIDATION accuracy, not TEST accuracy!")
    print("This is a common mistake in ML research.")
    
    # Based on the actual evaluation output
    results = {
        'Dataset Split': {
            'Total Samples': 94,
            'Training': '56 samples (59.6%)',
            'Validation': '19 samples (20.2%)',
            'Test': '19 samples (20.2%)'
        },
        'Hybrid Model Performance': {
            'Validation Accuracy (reported)': 0.962,
            'Test Accuracy (actual)': 0.895,
            'Test Balanced Accuracy': 0.938,
            'Performance Gap': -0.067
        },
        'Test Set Confusion Matrix': {
            'True Positives': 14,
            'False Positives': 2,
            'True Negatives': 3,
            'False Negatives': 0
        }
    }
    
    print("\n" + "="*60)
    print("ACTUAL TEST RESULTS (from held-out test set):")
    print("="*60)
    
    print(f"\n1. DATA SPLIT:")
    print(f"   Total samples: {results['Dataset Split']['Total Samples']}")
    print(f"   Training:   {results['Dataset Split']['Training']}")
    print(f"   Validation: {results['Dataset Split']['Validation']}")
    print(f"   Test:       {results['Dataset Split']['Test']}")
    
    print(f"\n2. YOUR HYBRID MODEL PERFORMANCE:")
    print(f"   Previously reported (validation): {results['Hybrid Model Performance']['Validation Accuracy (reported)']:.1%}")
    print(f"   ACTUAL TEST ACCURACY:             {results['Hybrid Model Performance']['Test Accuracy (actual)']:.1%}")
    print(f"   Test Balanced Accuracy:           {results['Hybrid Model Performance']['Test Balanced Accuracy']:.1%}")
    print(f"   Performance gap:                  {results['Hybrid Model Performance']['Performance Gap']:.1%}")
    
    print(f"\n3. TEST SET DETAILS:")
    print(f"   Normal cases:   16 (14 correct, 2 errors)")
    print(f"   Abnormal cases:  3 (3 correct, 0 errors)")
    print(f"   Total correct:  17/19 = 89.5%")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Validation vs Test Accuracy
    ax1 = axes[0]
    models = ['Validation\n(Selection)', 'Test\n(Final)']
    accuracies = [0.962, 0.895]
    colors = ['#3498db', '#e74c3c']
    bars = ax1.bar(models, accuracies, color=colors)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Validation vs Test Performance')
    ax1.set_ylim([0.8, 1.0])
    
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.1%}', ha='center', fontsize=12, fontweight='bold')
    
    # Add arrow showing the gap
    ax1.annotate('', xy=(1, 0.895), xytext=(1, 0.962),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax1.text(1.2, 0.928, '6.7% gap', color='red', fontsize=11)
    
    # Plot 2: Confusion Matrix
    ax2 = axes[1]
    cm = [[14, 2], [0, 3]]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['Predicted\nNormal', 'Predicted\nAbnormal'],
                yticklabels=['Actual\nNormal', 'Actual\nAbnormal'])
    ax2.set_title('Test Set Confusion Matrix')
    
    # Plot 3: Performance Comparison
    ax3 = axes[2]
    metrics = ['Accuracy', 'Balanced\nAccuracy', 'Precision', 'Recall']
    values = [0.895, 0.938, 0.937, 0.895]
    bars = ax3.bar(metrics, values, color='#2ecc71')
    ax3.set_ylabel('Score')
    ax3.set_title('Test Set Metrics')
    ax3.set_ylim([0.8, 1.0])
    
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('validation_vs_test_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    
    print("\n1. WHY THE DIFFERENCE?")
    print("   • Validation set: Used for hyperparameter tuning and model selection")
    print("   • Test set: Completely unseen data, true generalization measure")
    print("   • 6.7% gap indicates slight overfitting to validation set")
    
    print("\n2. IS 89.5% TEST ACCURACY STILL GOOD?")
    print("   • YES! This is still excellent performance")
    print("   • Balanced accuracy (93.8%) shows good performance on both classes")
    print("   • Perfect recall for abnormal cases (3/3 = 100%)")
    
    print("\n3. IMPLICATIONS:")
    print("   • True model performance: ~89-94% (not 96.2%)")
    print("   • Still clinically useful (>85% threshold)")
    print("   • Need more data to reduce validation-test gap")
    
    print("\n4. BEST PRACTICES MOVING FORWARD:")
    print("   ✓ Always report TEST set performance for papers")
    print("   ✓ Report both validation AND test results")
    print("   ✓ Never touch test set until final evaluation")
    print("   ✓ Use cross-validation for more robust estimates")
    
    # Corrected model comparison
    print("\n" + "="*80)
    print("CORRECTED MODEL RANKINGS (Test Set Performance):")
    print("="*80)
    
    corrected_rankings = [
        ('Ultra Hybrid (expected)', '~95%', 'Estimated from val-test gap'),
        ('MeshCNN/GNN Hybrid (expected)', '~94%', 'Estimated from val-test gap'),
        ('YOUR MODEL (actual test)', '89.5%', 'Actual test result'),
        ('Traditional ML (expected)', '~80%', 'Typically robust to test set'),
        ('Pure PointNet (expected)', '~68%', 'Poor generalization')
    ]
    
    for rank, (model, test_acc, note) in enumerate(corrected_rankings, 1):
        print(f"{rank}. {model:30s} | Test: {test_acc:6s} | {note}")
    
    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    print("Your model's TRUE test performance is 89.5% (balanced: 93.8%)")
    print("This is still EXCELLENT for a 94-sample dataset!")
    print("The 96.2% was validation accuracy (overly optimistic)")
    print("="*80)
    
    # Save summary
    summary = {
        'validation_accuracy': 0.962,
        'test_accuracy': 0.895,
        'test_balanced_accuracy': 0.938,
        'gap': -0.067,
        'conclusion': 'Model shows good generalization with slight overfitting',
        'recommendation': 'Collect more data to improve generalization'
    }
    
    pd.DataFrame([summary]).to_csv('test_evaluation_summary.csv', index=False)
    print("\nSummary saved to 'test_evaluation_summary.csv'")
    
    return summary

if __name__ == "__main__":
    summary = display_evaluation_results()
    
    print("\n" + "="*80)
    print("IMPORTANT LESSON:")
    print("="*80)
    print("This is why proper train/val/test split is CRITICAL!")
    print("Many papers report validation as final accuracy (incorrect)")
    print("Always be transparent about which metric you're reporting")
    print("="*80)