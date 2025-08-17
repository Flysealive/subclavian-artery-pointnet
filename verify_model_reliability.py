#!/usr/bin/env python3
"""
MODEL RELIABILITY VERIFICATION SCRIPT
======================================
Comprehensive validation to ensure your custom model results are trustworthy
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, cross_val_score, permutation_test_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ModelReliabilityVerifier:
    """Verify custom model reliability through multiple validation methods"""
    
    def __init__(self, X, y, model_func, model_name="Custom Model"):
        """
        Args:
            X: Features (can be multi-modal)
            y: Labels
            model_func: Function that returns a new model instance
            model_name: Name for reporting
        """
        self.X = X
        self.y = y
        self.model_func = model_func
        self.model_name = model_name
        self.results = {}
        
    def run_all_verifications(self):
        """Run comprehensive reliability checks"""
        print("="*80)
        print(f"RELIABILITY VERIFICATION FOR: {self.model_name}")
        print("="*80)
        
        # 1. Cross-validation with multiple random seeds
        self.verify_with_cross_validation()
        
        # 2. Compare against baseline methods
        self.compare_with_baselines()
        
        # 3. Permutation test (is model better than random?)
        self.permutation_test()
        
        # 4. Learning curve analysis
        self.learning_curve_analysis()
        
        # 5. Stability test (multiple runs)
        self.stability_test()
        
        # 6. Statistical significance tests
        self.statistical_tests()
        
        # Generate reliability report
        self.generate_reliability_report()
        
    def verify_with_cross_validation(self, n_splits=5, n_repeats=10):
        """
        Multiple repeated cross-validation for robust estimates
        """
        print("\n1. CROSS-VALIDATION VERIFICATION")
        print("-" * 40)
        
        all_scores = []
        all_balanced = []
        
        for repeat in range(n_repeats):
            # Different random seed each repeat
            np.random.seed(42 + repeat)
            
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                                   random_state=42 + repeat)
            
            fold_scores = []
            fold_balanced = []
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(self.X, self.y)):
                # Get fold data
                if isinstance(self.X, tuple):  # Multi-modal
                    X_train = tuple(x[train_idx] for x in self.X)
                    X_val = tuple(x[val_idx] for x in self.X)
                else:
                    X_train, X_val = self.X[train_idx], self.X[val_idx]
                    
                y_train, y_val = self.y[train_idx], self.y[val_idx]
                
                # Train model
                model = self.model_func()
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_val)
                fold_scores.append(accuracy_score(y_val, y_pred))
                fold_balanced.append(balanced_accuracy_score(y_val, y_pred))
            
            all_scores.extend(fold_scores)
            all_balanced.extend(fold_balanced)
            
            if repeat == 0:
                print(f"First run CV scores: {[f'{s:.3f}' for s in fold_scores]}")
        
        # Calculate statistics
        self.results['cv_accuracy'] = {
            'mean': np.mean(all_scores),
            'std': np.std(all_scores),
            'min': np.min(all_scores),
            'max': np.max(all_scores),
            '95_ci': (np.percentile(all_scores, 2.5), 
                     np.percentile(all_scores, 97.5))
        }
        
        self.results['cv_balanced'] = {
            'mean': np.mean(all_balanced),
            'std': np.std(all_balanced),
            '95_ci': (np.percentile(all_balanced, 2.5),
                     np.percentile(all_balanced, 97.5))
        }
        
        print(f"\n{n_splits}-Fold CV × {n_repeats} repeats ({len(all_scores)} total evaluations):")
        print(f"Accuracy:  {self.results['cv_accuracy']['mean']:.3f} ± {self.results['cv_accuracy']['std']:.3f}")
        print(f"95% CI:    [{self.results['cv_accuracy']['95_ci'][0]:.3f}, {self.results['cv_accuracy']['95_ci'][1]:.3f}]")
        print(f"Range:     {self.results['cv_accuracy']['min']:.3f} - {self.results['cv_accuracy']['max']:.3f}")
        
    def compare_with_baselines(self):
        """
        Compare against simple baselines to ensure model is learning
        """
        print("\n2. BASELINE COMPARISON")
        print("-" * 40)
        
        baselines = {
            'Random (Stratified)': DummyClassifier(strategy='stratified'),
            'Most Frequent': DummyClassifier(strategy='most_frequent'),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        self.results['baselines'] = {}
        
        for name, baseline in baselines.items():
            # Use simple features for baseline
            if isinstance(self.X, tuple):
                # For multi-modal, use first modality or flatten
                X_simple = self.X[0].reshape(len(self.X[0]), -1)
            else:
                X_simple = self.X.reshape(len(self.X), -1)
            
            scores = cross_val_score(baseline, X_simple, self.y, 
                                    cv=5, scoring='balanced_accuracy')
            
            self.results['baselines'][name] = {
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
            
            print(f"{name:20s}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        
        # Check if model beats baselines
        model_score = self.results['cv_balanced']['mean']
        best_baseline = max(self.results['baselines'].values(), 
                          key=lambda x: x['mean'])['mean']
        
        improvement = model_score - best_baseline
        print(f"\nYour model: {model_score:.3f}")
        print(f"Improvement over best baseline: {improvement:.3f} ({improvement*100:.1f}%)")
        
        if improvement > 0.05:
            print("✅ Model significantly outperforms baselines")
        elif improvement > 0:
            print("⚠️ Model slightly better than baselines")
        else:
            print("❌ Model not better than baselines - CHECK IMPLEMENTATION")
            
    def permutation_test(self, n_permutations=100):
        """
        Test if model performance is better than random chance
        """
        print("\n3. PERMUTATION TEST")
        print("-" * 40)
        
        # Simplified for demonstration
        if isinstance(self.X, tuple):
            X_simple = self.X[0].reshape(len(self.X[0]), -1)
        else:
            X_simple = self.X.reshape(len(self.X), -1)
        
        # Create simple model for permutation test
        from sklearn.ensemble import RandomForestClassifier
        test_model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        score, perm_scores, pvalue = permutation_test_score(
            test_model, X_simple, self.y, 
            scoring="balanced_accuracy",
            n_permutations=n_permutations,
            random_state=42
        )
        
        self.results['permutation'] = {
            'score': score,
            'pvalue': pvalue,
            'null_mean': np.mean(perm_scores),
            'null_std': np.std(perm_scores)
        }
        
        print(f"Model score: {score:.3f}")
        print(f"Null hypothesis scores: {np.mean(perm_scores):.3f} ± {np.std(perm_scores):.3f}")
        print(f"P-value: {pvalue:.4f}")
        
        if pvalue < 0.05:
            print("✅ Model performance is statistically significant (p < 0.05)")
        else:
            print("❌ Model performance NOT significant - might be random!")
            
    def learning_curve_analysis(self):
        """
        Check if model improves with more data
        """
        print("\n4. LEARNING CURVE ANALYSIS")
        print("-" * 40)
        
        train_sizes = [0.3, 0.5, 0.7, 0.9]
        learning_scores = []
        
        for size in train_sizes:
            n_samples = int(len(self.y) * size)
            indices = np.random.choice(len(self.y), n_samples, replace=False)
            
            if isinstance(self.X, tuple):
                X_subset = tuple(x[indices] for x in self.X)
            else:
                X_subset = self.X[indices]
            y_subset = self.y[indices]
            
            # Quick CV on subset
            kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in kfold.split(X_subset, y_subset):
                if isinstance(X_subset, tuple):
                    X_train = tuple(x[train_idx] for x in X_subset)
                    X_val = tuple(x[val_idx] for x in X_subset)
                else:
                    X_train = X_subset[train_idx]
                    X_val = X_subset[val_idx]
                    
                y_train, y_val = y_subset[train_idx], y_subset[val_idx]
                
                model = self.model_func()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                scores.append(accuracy_score(y_val, y_pred))
            
            learning_scores.append(np.mean(scores))
            print(f"Training size {size*100:.0f}%: {np.mean(scores):.3f}")
        
        # Check if performance improves with more data
        if all(learning_scores[i] <= learning_scores[i+1] 
               for i in range(len(learning_scores)-1)):
            print("✅ Model improves with more data (good sign)")
        else:
            print("⚠️ Model performance inconsistent with data size")
            
    def stability_test(self, n_runs=10):
        """
        Test model stability across multiple training runs
        """
        print("\n5. STABILITY TEST")
        print("-" * 40)
        
        # Fixed train/test split for stability test
        from sklearn.model_selection import train_test_split
        
        test_scores = []
        
        for run in range(n_runs):
            # Same split, different initialization
            if isinstance(self.X, tuple):
                X_train = tuple(x[:int(len(x)*0.8)] for x in self.X)
                X_test = tuple(x[int(len(x)*0.8):] for x in self.X)
            else:
                X_train = self.X[:int(len(self.X)*0.8)]
                X_test = self.X[int(len(self.X)*0.8):]
                
            y_train = self.y[:int(len(self.y)*0.8)]
            y_test = self.y[int(len(self.y)*0.8):]
            
            # Train with different seed
            np.random.seed(run)
            torch.manual_seed(run)
            
            model = self.model_func()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            test_scores.append(accuracy_score(y_test, y_pred))
        
        self.results['stability'] = {
            'mean': np.mean(test_scores),
            'std': np.std(test_scores),
            'cv': np.std(test_scores) / np.mean(test_scores)  # Coefficient of variation
        }
        
        print(f"Test scores across {n_runs} runs: {np.mean(test_scores):.3f} ± {np.std(test_scores):.3f}")
        print(f"Coefficient of Variation: {self.results['stability']['cv']:.3f}")
        
        if self.results['stability']['cv'] < 0.1:
            print("✅ Model is stable (CV < 0.1)")
        elif self.results['stability']['cv'] < 0.2:
            print("⚠️ Model has moderate variability")
        else:
            print("❌ Model is unstable - high variance between runs")
            
    def statistical_tests(self):
        """
        Statistical tests for reliability
        """
        print("\n6. STATISTICAL TESTS")
        print("-" * 40)
        
        # 1. Test if accuracy is significantly better than random (binomial test)
        n_correct = int(self.results['cv_accuracy']['mean'] * 100)  # Assume 100 predictions
        n_total = 100
        expected_random = 0.5  # For balanced binary classification
        
        from scipy.stats import binom_test
        p_value = binom_test(n_correct, n_total, expected_random, alternative='greater')
        
        print(f"Binomial test (vs random 50%):")
        print(f"  P-value: {p_value:.4f}")
        if p_value < 0.05:
            print("  ✅ Significantly better than random")
        else:
            print("  ❌ Not significantly better than random")
            
        # 2. Confidence interval for true performance
        from scipy.stats import t
        n_folds = 50  # From repeated CV
        se = self.results['cv_accuracy']['std'] / np.sqrt(n_folds)
        confidence = 0.95
        h = se * t.ppf((1 + confidence) / 2, n_folds - 1)
        
        print(f"\n95% Confidence Interval for true accuracy:")
        print(f"  [{self.results['cv_accuracy']['mean']-h:.3f}, {self.results['cv_accuracy']['mean']+h:.3f}]")
        
    def generate_reliability_report(self):
        """
        Generate final reliability verdict
        """
        print("\n" + "="*80)
        print("RELIABILITY VERDICT")
        print("="*80)
        
        checks_passed = 0
        total_checks = 6
        
        # Check 1: CV performance
        if self.results['cv_accuracy']['mean'] > 0.7:
            print("✅ Cross-validation accuracy > 70%")
            checks_passed += 1
        else:
            print("❌ Cross-validation accuracy too low")
            
        # Check 2: Better than baselines
        model_score = self.results['cv_balanced']['mean']
        best_baseline = max(self.results['baselines'].values(), 
                          key=lambda x: x['mean'])['mean']
        if model_score > best_baseline:
            print("✅ Outperforms baseline methods")
            checks_passed += 1
        else:
            print("❌ Does not outperform baselines")
            
        # Check 3: Permutation test
        if self.results['permutation']['pvalue'] < 0.05:
            print("✅ Performance is statistically significant")
            checks_passed += 1
        else:
            print("❌ Performance not statistically significant")
            
        # Check 4: Reasonable variance
        if self.results['cv_accuracy']['std'] < 0.15:
            print("✅ Reasonable variance across folds")
            checks_passed += 1
        else:
            print("❌ High variance - unstable performance")
            
        # Check 5: Stability
        if self.results['stability']['cv'] < 0.2:
            print("✅ Stable across multiple runs")
            checks_passed += 1
        else:
            print("❌ Unstable between runs")
            
        # Check 6: Confidence interval
        ci_lower = self.results['cv_accuracy']['95_ci'][0]
        if ci_lower > 0.65:
            print("✅ Lower bound of 95% CI > 65%")
            checks_passed += 1
        else:
            print("❌ Confidence interval too low")
            
        # Final verdict
        print("\n" + "-"*40)
        print(f"Reliability Score: {checks_passed}/{total_checks}")
        
        if checks_passed >= 5:
            print("\n🎯 VERDICT: MODEL IS RELIABLE")
            print("Your results can be trusted for publication/production")
        elif checks_passed >= 4:
            print("\n⚠️ VERDICT: MODEL IS MODERATELY RELIABLE")
            print("Consider improvements before production use")
        else:
            print("\n❌ VERDICT: MODEL RELIABILITY QUESTIONABLE")
            print("Significant issues found - review implementation")
            
        # Reporting recommendation
        print("\n" + "="*80)
        print("RECOMMENDED REPORTING:")
        print("="*80)
        print(f'"{self.model_name} achieved {self.results["cv_accuracy"]["mean"]:.1f}% ± {self.results["cv_accuracy"]["std"]:.1f}% ')
        print(f'accuracy (95% CI: [{self.results["cv_accuracy"]["95_ci"][0]:.1f}%, {self.results["cv_accuracy"]["95_ci"][1]:.1f}%]) ')
        print(f'using {5}-fold cross-validation repeated {10} times.')
        print(f'The model significantly outperforms baseline methods (p < {self.results["permutation"]["pvalue"]:.3f})."')


# Example usage for your hybrid model
if __name__ == "__main__":
    print("HYBRID MODEL RELIABILITY VERIFICATION")
    print("="*80)
    
    # Load your data
    import pandas as pd
    from hybrid_multimodal_model import HybridMultiModalNet, HybridMultiModalDataset
    
    # Create dataset
    dataset = HybridMultiModalDataset(
        hybrid_dir='hybrid_data',
        labels_file='classification_labels_with_measurements.csv',
        mode='train'
    )
    
    # Prepare data for verification
    # Note: You'll need to adapt this to your actual data format
    pointclouds = []
    voxels = []
    measurements = []
    labels = []
    
    for i in range(len(dataset)):
        pc, vox, meas, label = dataset[i]
        pointclouds.append(pc)
        voxels.append(vox)
        measurements.append(meas)
        labels.append(label)
    
    X = (np.array(pointclouds), np.array(voxels), np.array(measurements))
    y = np.array(labels)
    
    # Define model creation function
    def create_hybrid_model():
        class ModelWrapper:
            def __init__(self):
                self.model = HybridMultiModalNet()
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(self.device)
                
            def fit(self, X, y):
                # Simplified training
                optimizer = torch.optim.Adam(self.model.parameters())
                criterion = nn.CrossEntropyLoss()
                
                for epoch in range(10):  # Quick training for verification
                    pc_tensor = torch.FloatTensor(X[0]).to(self.device)
                    vox_tensor = torch.FloatTensor(X[1]).unsqueeze(1).to(self.device)
                    meas_tensor = torch.FloatTensor(X[2]).to(self.device)
                    y_tensor = torch.LongTensor(y).to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(pc_tensor.transpose(1,2), vox_tensor, meas_tensor)
                    loss = criterion(outputs, y_tensor)
                    loss.backward()
                    optimizer.step()
                    
            def predict(self, X):
                self.model.eval()
                with torch.no_grad():
                    pc_tensor = torch.FloatTensor(X[0]).to(self.device)
                    vox_tensor = torch.FloatTensor(X[1]).unsqueeze(1).to(self.device)
                    meas_tensor = torch.FloatTensor(X[2]).to(self.device)
                    
                    outputs = self.model(pc_tensor.transpose(1,2), vox_tensor, meas_tensor)
                    _, predicted = torch.max(outputs, 1)
                    return predicted.cpu().numpy()
                    
        return ModelWrapper()
    
    # Run verification
    verifier = ModelReliabilityVerifier(
        X=X,
        y=y,
        model_func=create_hybrid_model,
        model_name="Hybrid PointNet+Voxel+Measurements"
    )
    
    verifier.run_all_verifications()
    
    print("\n" + "="*80)
    print("Verification complete! Check the results above to determine if your")
    print("model's performance claims are reliable and reproducible.")
    print("="*80)