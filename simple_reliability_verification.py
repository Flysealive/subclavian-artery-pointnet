#!/usr/bin/env python3
"""
SIMPLE MODEL RELIABILITY VERIFICATION
======================================
Tests model reliability using traditional ML for demonstration
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (StratifiedKFold, cross_val_score, 
                                   train_test_split, learning_curve)
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                           confusion_matrix, classification_report)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load STL features and labels"""
    print("Loading data...")
    
    # Try to load existing features if available
    import os
    import pickle
    
    if os.path.exists('stl_features.pkl'):
        print("Loading pre-computed features...")
        with open('stl_features.pkl', 'rb') as f:
            features_df = pickle.load(f)
    else:
        print("Computing features from STL files...")
        # Import feature extraction
        import trimesh
        from glob import glob
        
        stl_files = glob('STL/*.stl')
        features_list = []
        
        for stl_file in stl_files:
            try:
                mesh = trimesh.load(stl_file)
                
                # Extract basic geometric features
                features = {
                    'filename': os.path.basename(stl_file).replace('.stl', ''),
                    'volume': mesh.volume,
                    'surface_area': mesh.area,
                    'num_vertices': len(mesh.vertices),
                    'num_faces': len(mesh.faces),
                    'bbox_volume': mesh.bounding_box.volume,
                    'bbox_longest_axis': np.max(mesh.extents),
                    'bbox_shortest_axis': np.min(mesh.extents),
                    'compactness': (36 * np.pi * mesh.volume**2)**(1/3) / mesh.area,
                    'sphericity': (np.pi**(1/3) * (6*mesh.volume)**(2/3)) / mesh.area,
                    'extent_ratio': np.max(mesh.extents) / np.min(mesh.extents),
                    'mean_edge_length': np.mean(mesh.edges_unique_length),
                    'std_edge_length': np.std(mesh.edges_unique_length),
                }
                
                # Add vertex statistics
                vertices = mesh.vertices
                for i, axis in enumerate(['x', 'y', 'z']):
                    features[f'vertex_{axis}_mean'] = np.mean(vertices[:, i])
                    features[f'vertex_{axis}_std'] = np.std(vertices[:, i])
                    features[f'vertex_{axis}_min'] = np.min(vertices[:, i])
                    features[f'vertex_{axis}_max'] = np.max(vertices[:, i])
                
                features_list.append(features)
                
            except Exception as e:
                print(f"Error processing {stl_file}: {e}")
        
        features_df = pd.DataFrame(features_list)
        
        # Save for future use
        with open('stl_features.pkl', 'wb') as f:
            pickle.dump(features_df, f)
    
    # Load labels and measurements
    labels_df = pd.read_csv('classification_labels_with_measurements.csv')
    
    # Merge features with labels
    merged_df = pd.merge(features_df, labels_df, on='filename', how='inner')
    
    # Prepare features
    feature_columns = [col for col in merged_df.columns 
                      if col not in ['filename', 'label']]
    
    X = merged_df[feature_columns].values
    y = merged_df['label'].values
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0)
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, merged_df['filename'].values

def comprehensive_reliability_test(X, y, model_name="Random Forest"):
    """Run comprehensive reliability tests"""
    
    print("\n" + "="*80)
    print(f"RELIABILITY VERIFICATION: {model_name}")
    print("="*80)
    
    results = {}
    
    # 1. REPEATED CROSS-VALIDATION
    print("\n1. REPEATED CROSS-VALIDATION (5-fold × 10 repeats)")
    print("-" * 50)
    
    all_scores = []
    all_balanced = []
    all_predictions = []
    all_true = []
    
    for repeat in range(10):
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42+repeat)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            if model_name == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42+repeat)
            else:
                model = GradientBoostingClassifier(n_estimators=100, random_state=42+repeat)
            
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            acc = accuracy_score(y_val, y_pred)
            balanced = balanced_accuracy_score(y_val, y_pred)
            
            all_scores.append(acc)
            all_balanced.append(balanced)
            all_predictions.extend(y_pred)
            all_true.extend(y_val)
            
            if repeat == 0:  # Print first repeat
                print(f"  Fold {fold+1}: Acc={acc:.3f}, Balanced={balanced:.3f}")
    
    # Statistics
    mean_acc = np.mean(all_scores)
    std_acc = np.std(all_scores)
    ci_lower = np.percentile(all_scores, 2.5)
    ci_upper = np.percentile(all_scores, 97.5)
    
    print(f"\nResults from {len(all_scores)} evaluations:")
    print(f"  Accuracy:  {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"  95% CI:    [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"  Range:     {np.min(all_scores):.3f} - {np.max(all_scores):.3f}")
    
    results['cv_accuracy'] = mean_acc
    results['cv_std'] = std_acc
    results['cv_ci'] = (ci_lower, ci_upper)
    
    # 2. BASELINE COMPARISON
    print("\n2. BASELINE COMPARISON")
    print("-" * 50)
    
    baselines = {
        'Random (Stratified)': DummyClassifier(strategy='stratified', random_state=42),
        'Most Frequent': DummyClassifier(strategy='most_frequent'),
        'Prior (Always Normal)': DummyClassifier(strategy='constant', constant=0),
    }
    
    for name, baseline in baselines.items():
        scores = cross_val_score(baseline, X, y, cv=5, scoring='balanced_accuracy')
        print(f"  {name:25s}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        
        if name == 'Random (Stratified)':
            random_score = np.mean(scores)
    
    improvement = mean_acc - random_score
    print(f"\nYour model improvement over random: {improvement:.3f} ({improvement*100:.1f}%)")
    
    if improvement > 0.2:
        print("✅ Model significantly outperforms random baseline")
    elif improvement > 0.1:
        print("⚠️ Model moderately outperforms random baseline")
    else:
        print("❌ Model barely better than random!")
    
    results['improvement_over_random'] = improvement
    
    # 3. LEARNING CURVE
    print("\n3. LEARNING CURVE ANALYSIS")
    print("-" * 50)
    
    train_sizes = [0.3, 0.5, 0.7, 0.9]
    learning_scores = []
    
    for size in train_sizes:
        # Use subset of data
        n_samples = int(len(y) * size)
        indices = np.random.choice(len(y), n_samples, replace=False)
        X_subset = X[indices]
        y_subset = y[indices]
        
        # Quick CV
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        scores = cross_val_score(model, X_subset, y_subset, cv=3)
        learning_scores.append(np.mean(scores))
        print(f"  Training size {size*100:3.0f}%: {np.mean(scores):.3f}")
    
    # Check if improves with more data
    if all(learning_scores[i] <= learning_scores[i+1] for i in range(len(learning_scores)-1)):
        print("✅ Performance improves with more data (good sign)")
    else:
        print("⚠️ Performance inconsistent with data size")
    
    # 4. STABILITY TEST
    print("\n4. STABILITY TEST (10 runs, same split)")
    print("-" * 50)
    
    # Fixed split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    stability_scores = []
    for run in range(10):
        model = RandomForestClassifier(n_estimators=100, random_state=run)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        stability_scores.append(accuracy_score(y_test, y_pred))
    
    stability_mean = np.mean(stability_scores)
    stability_std = np.std(stability_scores)
    cv_coefficient = stability_std / stability_mean if stability_mean > 0 else 1
    
    print(f"  Scores: {[f'{s:.3f}' for s in stability_scores]}")
    print(f"  Mean ± Std: {stability_mean:.3f} ± {stability_std:.3f}")
    print(f"  Coefficient of Variation: {cv_coefficient:.3f}")
    
    if cv_coefficient < 0.05:
        print("✅ Very stable model (CV < 0.05)")
    elif cv_coefficient < 0.1:
        print("✅ Stable model (CV < 0.1)")
    else:
        print("⚠️ Some instability detected")
    
    results['stability_cv'] = cv_coefficient
    
    # 5. STATISTICAL SIGNIFICANCE
    print("\n5. STATISTICAL SIGNIFICANCE TESTS")
    print("-" * 50)
    
    # Binomial test
    n_correct = int(mean_acc * 100)
    n_total = 100
    from scipy.stats import binom_test
    p_value = binom_test(n_correct, n_total, 0.5, alternative='greater')
    
    print(f"  Binomial test (vs 50% random):")
    print(f"    P-value: {p_value:.6f}")
    if p_value < 0.001:
        print("    ✅ Highly significant (p < 0.001)")
    elif p_value < 0.05:
        print("    ✅ Significant (p < 0.05)")
    else:
        print("    ❌ Not significant")
    
    results['p_value'] = p_value
    
    # 6. CONFUSION MATRIX ANALYSIS
    print("\n6. DETAILED PERFORMANCE METRICS")
    print("-" * 50)
    
    # Train final model for detailed analysis
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"            Neg    Pos")
    print(f"Actual Neg  {cm[0,0]:3d}    {cm[0,1]:3d}")
    print(f"       Pos  {cm[1,0]:3d}    {cm[1,1]:3d}")
    
    # Per-class accuracy
    if cm[0,0] + cm[0,1] > 0:
        class0_acc = cm[0,0] / (cm[0,0] + cm[0,1])
        print(f"\nClass 0 (Normal) accuracy: {class0_acc:.3f}")
    if cm[1,0] + cm[1,1] > 0:
        class1_acc = cm[1,1] / (cm[1,0] + cm[1,1])
        print(f"Class 1 (Abnormal) accuracy: {class1_acc:.3f}")
    
    # Feature importance (for Random Forest)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_features_idx = np.argsort(importances)[-5:][::-1]
        print("\nTop 5 Important Features:")
        feature_names = ['volume', 'surface_area', 'num_vertices', 'num_faces', 
                        'bbox_volume', 'bbox_longest_axis', 'bbox_shortest_axis',
                        'compactness', 'sphericity', 'extent_ratio', 
                        'mean_edge_length', 'std_edge_length'] + \
                       [f'vertex_{ax}_{stat}' for ax in ['x','y','z'] 
                        for stat in ['mean','std','min','max']] + \
                       ['left_subclavian_diameter_mm', 'aortic_arch_diameter_mm', 'angle_degrees']
        
        for i, idx in enumerate(top_features_idx):
            if idx < len(feature_names):
                print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.3f}")
    
    # FINAL VERDICT
    print("\n" + "="*80)
    print("RELIABILITY VERDICT")
    print("="*80)
    
    checks_passed = 0
    total_checks = 6
    
    # Check criteria
    if mean_acc > 0.7:
        print("✅ Cross-validation accuracy > 70%")
        checks_passed += 1
    else:
        print("❌ Cross-validation accuracy < 70%")
    
    if improvement > 0.2:
        print("✅ Significantly better than random")
        checks_passed += 1
    else:
        print("❌ Not much better than random")
    
    if p_value < 0.05:
        print("✅ Statistically significant (p < 0.05)")
        checks_passed += 1
    else:
        print("❌ Not statistically significant")
    
    if std_acc < 0.15:
        print("✅ Low variance across folds (< 15%)")
        checks_passed += 1
    else:
        print("❌ High variance across folds")
    
    if cv_coefficient < 0.1:
        print("✅ Stable across multiple runs")
        checks_passed += 1
    else:
        print("❌ Unstable between runs")
    
    if ci_lower > 0.65:
        print("✅ 95% CI lower bound > 65%")
        checks_passed += 1
    else:
        print("❌ Confidence interval too low")
    
    print(f"\nReliability Score: {checks_passed}/{total_checks}")
    
    if checks_passed >= 5:
        verdict = "🎯 MODEL IS RELIABLE"
        verdict_detail = "Results can be trusted for publication/production"
    elif checks_passed >= 4:
        verdict = "⚠️ MODEL IS MODERATELY RELIABLE"
        verdict_detail = "Consider improvements before production"
    else:
        verdict = "❌ MODEL RELIABILITY QUESTIONABLE"
        verdict_detail = "Significant issues found - review approach"
    
    print(f"\n{verdict}")
    print(verdict_detail)
    
    # Recommended reporting
    print("\n" + "="*80)
    print("RECOMMENDED SCIENTIFIC REPORTING:")
    print("="*80)
    print(f'"{model_name} achieved {mean_acc*100:.1f}% ± {std_acc*100:.1f}% accuracy')
    print(f'(95% CI: [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]) using 5-fold')
    print(f'cross-validation repeated 10 times (n={len(X)} samples).')
    print(f'The model significantly outperforms random classification (p<{p_value:.3f})."')
    
    return results

def main():
    """Main execution"""
    print("="*80)
    print("MODEL RELIABILITY VERIFICATION SYSTEM")
    print("="*80)
    
    # Load data
    X, y, filenames = load_and_prepare_data()
    
    # Test Random Forest (simulating your hybrid model)
    print("\nTesting Random Forest (Traditional ML Baseline)...")
    rf_results = comprehensive_reliability_test(X, y, "Random Forest")
    
    # Test Gradient Boosting
    print("\n\nTesting Gradient Boosting...")
    gb_results = comprehensive_reliability_test(X, y, "Gradient Boosting")
    
    # Summary comparison
    print("\n" + "="*80)
    print("COMPARATIVE SUMMARY")
    print("="*80)
    print(f"Random Forest:        {rf_results['cv_accuracy']:.3f} ± {rf_results['cv_std']:.3f}")
    print(f"Gradient Boosting:    {gb_results['cv_accuracy']:.3f} ± {gb_results['cv_std']:.3f}")
    print(f"\nBest model: {'Random Forest' if rf_results['cv_accuracy'] > gb_results['cv_accuracy'] else 'Gradient Boosting'}")
    
    print("\n" + "="*80)
    print("WHAT THIS MEANS FOR YOUR HYBRID MODEL:")
    print("="*80)
    print("Your reported 96.2% validation / 89.5% test accuracy should be verified by:")
    print("1. Running this same verification on your actual hybrid model")
    print("2. Using 5-fold cross-validation instead of single split")
    print("3. Reporting confidence intervals, not just point estimates")
    print("4. Comparing against these traditional ML baselines")
    print("\nIf your hybrid model shows similar stability (low variance) and")
    print("significantly outperforms these baselines, then it's reliable!")
    print("="*80)

if __name__ == "__main__":
    main()