#!/usr/bin/env python3
"""
CROSS-VALIDATION DEMONSTRATION
===============================
Shows what proper cross-validation would look like
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def simulate_cross_validation():
    """
    Simulate cross-validation to show what we should have done
    """
    print("="*80)
    print("CROSS-VALIDATION: WHAT WE SHOULD HAVE DONE")
    print("="*80)
    
    # Create simulated data based on your actual dataset
    np.random.seed(42)
    n_samples = 94
    n_features = 100  # Simulating extracted features
    
    # Create imbalanced classes like your data
    y = np.array([0]*78 + [1]*16)  # 78 normal, 16 abnormal
    np.random.shuffle(y)
    
    # Simulate feature matrix
    X = np.random.randn(n_samples, n_features)
    # Make features somewhat predictive
    X[y==1] += np.random.randn(n_features) * 0.5
    
    print(f"Dataset: {n_samples} samples, {n_features} features")
    print(f"Class distribution: Normal={np.sum(y==0)}, Abnormal={np.sum(y==1)}")
    print(f"Class imbalance: {np.sum(y==0)/np.sum(y==1):.1f}:1")
    
    # Compare different validation strategies
    strategies = {
        'Single Split (What we did)': None,
        '3-Fold CV': StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        '5-Fold CV (Recommended)': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        '10-Fold CV': StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        'Leave-One-Out': None  # Too expensive for deep learning
    }
    
    results = {}
    
    print("\n" + "="*60)
    print("VALIDATION STRATEGY COMPARISON")
    print("="*60)
    
    # 1. Single Split (What we did)
    print("\n1. SINGLE SPLIT (What we actually did):")
    print("-"*40)
    
    from sklearn.model_selection import train_test_split
    
    single_scores = []
    for seed in range(10):  # Try 10 different splits
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )
        
        # Simulate model training and prediction
        # In reality, this would be your neural network
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        score = balanced_accuracy_score(y_test, model.predict(X_test))
        single_scores.append(score)
    
    print(f"  10 different random splits:")
    print(f"  Scores: {[f'{s:.3f}' for s in single_scores]}")
    print(f"  Mean: {np.mean(single_scores):.3f}")
    print(f"  Std: {np.std(single_scores):.3f}")
    print(f"  Range: [{min(single_scores):.3f}, {max(single_scores):.3f}]")
    print(f"  Variability: {(max(single_scores)-min(single_scores)):.3f}")
    
    results['Single Split'] = {
        'scores': single_scores,
        'mean': np.mean(single_scores),
        'std': np.std(single_scores)
    }
    
    # 2. K-Fold Cross-Validation
    for name, cv in strategies.items():
        if cv is not None and 'Fold' in name:
            print(f"\n2. {name}:")
            print("-"*40)
            
            fold_scores = []
            fold_details = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train model
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                score = balanced_accuracy_score(y_val, model.predict(X_val))
                
                fold_scores.append(score)
                fold_details.append({
                    'fold': fold_idx,
                    'train_size': len(train_idx),
                    'val_size': len(val_idx),
                    'score': score
                })
            
            print(f"  Fold scores: {[f'{s:.3f}' for s in fold_scores]}")
            print(f"  Mean: {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}")
            print(f"  95% CI: [{np.mean(fold_scores)-1.96*np.std(fold_scores):.3f}, "
                  f"{np.mean(fold_scores)+1.96*np.std(fold_scores):.3f}]")
            print(f"  Variability: {(max(fold_scores)-min(fold_scores)):.3f}")
            
            results[name] = {
                'scores': fold_scores,
                'mean': np.mean(fold_scores),
                'std': np.std(fold_scores),
                'details': fold_details
            }
    
    return results

def visualize_cv_comparison(results):
    """Create visualization of CV comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Score distributions
    ax1 = axes[0, 0]
    data_to_plot = []
    labels = []
    
    for name, data in results.items():
        if 'scores' in data:
            data_to_plot.append(data['scores'])
            labels.append(name.replace(' (What we did)', '').replace(' (Recommended)', ''))
    
    bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], sns.color_palette('husl', len(labels))):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Balanced Accuracy')
    ax1.set_title('Score Distribution by Validation Strategy')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    
    # 2. Mean and Std comparison
    ax2 = axes[0, 1]
    means = []
    stds = []
    names = []
    
    for name, data in results.items():
        if 'mean' in data:
            means.append(data['mean'])
            stds.append(data['std'])
            names.append(name.split(' ')[0])
    
    x = np.arange(len(means))
    bars = ax2.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Mean Balanced Accuracy')
    ax2.set_title('Mean Performance ± Std Dev')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.3f}', ha='center', fontsize=9)
    
    # 3. Variability analysis
    ax3 = axes[1, 0]
    variabilities = []
    names_var = []
    
    for name, data in results.items():
        if 'scores' in data:
            var = max(data['scores']) - min(data['scores'])
            variabilities.append(var)
            names_var.append(name.split(' ')[0])
    
    bars = ax3.bar(names_var, variabilities, color='coral', alpha=0.7)
    ax3.set_ylabel('Score Range (Max - Min)')
    ax3.set_title('Variability of Results')
    ax3.set_xticklabels(names_var, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Sample efficiency
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    text = """
    CROSS-VALIDATION INSIGHTS:
    
    1. Single Split (What we did):
       • High variability between splits
       • No confidence intervals
       • Results depend on random seed
       • Can be overly optimistic or pessimistic
    
    2. K-Fold Cross-Validation (Recommended):
       • More stable estimates
       • Provides confidence intervals
       • Uses all data for validation
       • Standard practice for small datasets
    
    3. For Your 94 Samples:
       • 5-Fold CV is optimal
       • Each fold: ~75 train, ~19 validation
       • Similar to your 80/20 split but 5 times
    
    4. Impact on Reported Results:
       • Your 96.2% → Likely ~92% ± 4% with CV
       • More honest and reproducible
       • Required by many journals now
    """
    
    ax4.text(0.1, 0.9, text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle('Cross-Validation vs Single Split Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cross_validation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def show_deep_learning_cv():
    """Show how CV should work for deep learning"""
    
    print("\n" + "="*80)
    print("CROSS-VALIDATION FOR DEEP LEARNING")
    print("="*80)
    
    print("\nPROPER APPROACH FOR YOUR HYBRID MODEL:")
    print("-"*40)
    
    print("""
    for fold in range(5):
        # 1. Split data
        train_idx, val_idx = get_fold_indices(fold)
        
        # 2. Create data loaders
        train_loader = DataLoader(data[train_idx])
        val_loader = DataLoader(data[val_idx])
        
        # 3. Initialize NEW model for each fold
        model = HybridModel()  # Fresh initialization
        
        # 4. Train model
        for epoch in range(150):
            train(model, train_loader)
            val_score = validate(model, val_loader)
            
        # 5. Save fold results
        fold_scores.append(val_score)
    
    # 6. Report mean ± std
    print(f"Performance: {mean(fold_scores)} ± {std(fold_scores)}")
    """)
    
    print("\nTIME COST:")
    print("-"*40)
    print("  Single training: 1x time")
    print("  5-Fold CV: 5x time")
    print("  But you get: Confidence intervals + Robust estimates")
    
    print("\nMEMORY COST:")
    print("-"*40)
    print("  Same as single training (one model at a time)")
    
    print("\nBENEFITS:")
    print("-"*40)
    print("  ✓ No cherry-picking lucky splits")
    print("  ✓ Realistic performance estimates")
    print("  ✓ Can report confidence intervals")
    print("  ✓ More credible research")

def main():
    """Main execution"""
    
    print("="*80)
    print("CROSS-VALIDATION ANALYSIS")
    print("="*80)
    print("Demonstrating proper validation methodology")
    print("="*80)
    
    # Run simulation
    results = simulate_cross_validation()
    
    # Create visualizations
    visualize_cv_comparison(results)
    
    # Show deep learning specific
    show_deep_learning_cv()
    
    # Final summary
    print("\n" + "="*80)
    print("CRITICAL SUMMARY")
    print("="*80)
    
    print("\n❌ WHAT YOU DID:")
    print("  • Single 60/20/20 split")
    print("  • Trained once")
    print("  • Reported single number (96.2%)")
    print("  • No confidence intervals")
    
    print("\n✓ WHAT YOU SHOULD HAVE DONE:")
    print("  • 5-Fold cross-validation")
    print("  • Train 5 times")
    print("  • Report mean ± std (e.g., 92% ± 4%)")
    print("  • Include confidence intervals")
    
    print("\n📊 EXPECTED IMPACT:")
    print("  • Your 96.2% → ~92% ± 4%")
    print("  • Lower but more honest")
    print("  • More reproducible")
    print("  • More credible")
    
    print("\n💡 RECOMMENDATION:")
    print("  For your paper, either:")
    print("  1. Re-run with 5-fold CV (best)")
    print("  2. Acknowledge limitation: 'Due to computational")
    print("     constraints, we used a single train/val/test split'")
    
    print("\n" + "="*80)
    
    return results

if __name__ == "__main__":
    results = main()