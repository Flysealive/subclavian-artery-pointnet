#!/usr/bin/env python3
"""
WEIGHTED ENSEMBLE WITH OPTIMIZATION
====================================
Combines multiple models with optimized weights
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                           f1_score, confusion_matrix, classification_report)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class WeightedEnsemble:
    """Weighted ensemble with optimized weights"""
    
    def __init__(self):
        self.models = []
        self.weights = None
        
    def add_model(self, model, name):
        """Add a model to the ensemble"""
        self.models.append((name, model))
        
    def optimize_weights(self, X_val, y_val):
        """Find optimal weights for ensemble"""
        n_models = len(self.models)
        
        # Get predictions from all models
        predictions = []
        for name, model in self.models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_val)[:, 1]
            else:
                pred = model.predict(X_val)
            predictions.append(pred)
        predictions = np.array(predictions).T
        
        # Objective function to minimize
        def objective(weights):
            # Ensure weights sum to 1
            weights = weights / np.sum(weights)
            # Weighted average prediction
            weighted_pred = np.dot(predictions, weights)
            # Convert to class predictions
            y_pred = (weighted_pred >= 0.5).astype(int)
            # Return negative accuracy (we want to maximize accuracy)
            return -balanced_accuracy_score(y_val, y_pred)
        
        # Initial weights (equal)
        initial_weights = np.ones(n_models) / n_models
        
        # Constraints: weights sum to 1, all weights >= 0
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Optimize
        result = minimize(objective, initial_weights, 
                         method='SLSQP', bounds=bounds, 
                         constraints=constraints)
        
        self.weights = result.x
        return self.weights
    
    def predict(self, X):
        """Make weighted predictions"""
        if self.weights is None:
            # Use equal weights if not optimized
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        # Get predictions from all models
        predictions = []
        for name, model in self.models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            predictions.append(pred)
        predictions = np.array(predictions).T
        
        # Weighted average
        weighted_pred = np.dot(predictions, self.weights)
        
        # Convert to class predictions
        return (weighted_pred >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """Get probability predictions"""
        if self.weights is None:
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        # Get predictions from all models
        predictions = []
        for name, model in self.models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            predictions.append(pred)
        predictions = np.array(predictions).T
        
        # Weighted average
        weighted_pred = np.dot(predictions, self.weights)
        
        # Return as probability array
        proba = np.zeros((len(X), 2))
        proba[:, 0] = 1 - weighted_pred
        proba[:, 1] = weighted_pred
        return proba

def main():
    """Run weighted ensemble"""
    print("="*60)
    print("WEIGHTED ENSEMBLE OPTIMIZATION")
    print("="*60)
    
    # Load features
    print("\n1. Loading data...")
    with open('stl_features.pkl', 'rb') as f:
        features_dict = pickle.load(f)
    
    labels_df = pd.read_csv('classification_labels_with_measurements.csv')
    labels_df['filename_clean'] = labels_df['filename'].str.replace('.stl', '').str.replace('.npy', '')
    
    # Prepare data
    X_list = []
    y_list = []
    
    for _, row in labels_df.iterrows():
        filename = row['filename_clean']
        if filename in features_dict:
            features = features_dict[filename]
            measurements = row[['left_subclavian_diameter_mm', 
                              'aortic_arch_diameter_mm', 
                              'angle_degrees']].values
            combined_features = np.concatenate([features, measurements])
            X_list.append(combined_features)
            y_list.append(row['label'])
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Data shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split train into train and validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    
    print("\n2. Training individual models...")
    
    # Model 1: Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train_final, y_train_final)
    rf_acc = accuracy_score(y_val, rf.predict(X_val))
    print(f"  Random Forest Val Acc: {rf_acc:.3f}")
    
    # Model 2: Extra Trees
    et = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    et.fit(X_train_final, y_train_final)
    et_acc = accuracy_score(y_val, et.predict(X_val))
    print(f"  Extra Trees Val Acc: {et_acc:.3f}")
    
    # Model 3: Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb.fit(X_train_final, y_train_final)
    gb_acc = accuracy_score(y_val, gb.predict(X_val))
    print(f"  Gradient Boosting Val Acc: {gb_acc:.3f}")
    
    # Model 4: Another Random Forest with different params
    rf2 = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=3,
        class_weight='balanced_subsample',
        random_state=123
    )
    rf2.fit(X_train_final, y_train_final)
    rf2_acc = accuracy_score(y_val, rf2.predict(X_val))
    print(f"  Random Forest 2 Val Acc: {rf2_acc:.3f}")
    
    print("\n3. Creating weighted ensemble...")
    
    # Create ensemble
    ensemble = WeightedEnsemble()
    ensemble.add_model(rf, 'RandomForest')
    ensemble.add_model(et, 'ExtraTrees')
    ensemble.add_model(gb, 'GradientBoosting')
    ensemble.add_model(rf2, 'RandomForest2')
    
    # Optimize weights on validation set
    weights = ensemble.optimize_weights(X_val, y_val)
    
    print("\nOptimized weights:")
    for i, (name, _) in enumerate(ensemble.models):
        print(f"  {name}: {weights[i]:.3f}")
    
    print("\n4. Evaluating on test set...")
    
    # Individual model performances
    print("\nIndividual models on test set:")
    for name, model in ensemble.models:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        print(f"  {name:20s}: Acc={acc:.3f}, Bal_Acc={bal_acc:.3f}")
    
    # Ensemble performance
    y_pred_ensemble = ensemble.predict(X_test)
    ensemble_acc = accuracy_score(y_test, y_pred_ensemble)
    ensemble_bal_acc = balanced_accuracy_score(y_test, y_pred_ensemble)
    ensemble_f1 = f1_score(y_test, y_pred_ensemble, average='weighted')
    
    print("\nWEIGHTED ENSEMBLE on test set:")
    print(f"  Accuracy:          {ensemble_acc:.3f}")
    print(f"  Balanced Accuracy: {ensemble_bal_acc:.3f}")
    print(f"  F1-Score:          {ensemble_f1:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_ensemble)
    print("\nConfusion Matrix:")
    print("              Predicted")
    print("            Normal  Abnormal")
    print(f"Actual Normal  {cm[0,0]:2d}      {cm[0,1]:2d}")
    print(f"      Abnormal  {cm[1,0]:2d}      {cm[1,1]:2d}")
    
    # Improvement calculation
    best_individual = max([accuracy_score(y_test, model.predict(X_test)) 
                          for _, model in ensemble.models])
    improvement = ensemble_acc - best_individual
    
    print(f"\nImprovement over best individual: {improvement:+.3f} ({improvement*100:+.1f}%)")
    
    # Save ensemble
    with open('weighted_ensemble.pkl', 'wb') as f:
        pickle.dump(ensemble, f)
    print("\nEnsemble saved to weighted_ensemble.pkl")
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Weighted ensemble achieved {ensemble_acc*100:.1f}% accuracy")
    print(f"This represents a {improvement*100:+.1f}% improvement")
    print("over the best individual model")

if __name__ == "__main__":
    main()