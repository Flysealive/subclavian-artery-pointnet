"""
Traditional Machine Learning Approach with Hand-Crafted 3D Features
====================================================================
This approach extracts meaningful geometric features from STL files
and uses traditional ML algorithms that work better with small datasets.
"""

import numpy as np
import pandas as pd
import trimesh
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============== PART 1: FEATURE EXTRACTION ==============

class GeometricFeatureExtractor:
    """Extract meaningful geometric features from 3D meshes"""
    
    @staticmethod
    def extract_basic_features(mesh):
        """Extract basic geometric properties"""
        features = {}
        
        # Volume and surface area
        features['volume'] = mesh.volume
        features['surface_area'] = mesh.area
        features['volume_to_surface_ratio'] = mesh.volume / mesh.area if mesh.area > 0 else 0
        
        # Bounding box features
        bounds = mesh.bounds
        bbox_dims = bounds[1] - bounds[0]
        features['bbox_volume'] = np.prod(bbox_dims)
        features['bbox_x'] = bbox_dims[0]
        features['bbox_y'] = bbox_dims[1]
        features['bbox_z'] = bbox_dims[2]
        features['bbox_diagonal'] = np.linalg.norm(bbox_dims)
        features['compactness'] = mesh.volume / features['bbox_volume'] if features['bbox_volume'] > 0 else 0
        
        # Aspect ratios
        sorted_dims = np.sort(bbox_dims)
        features['aspect_ratio_1'] = sorted_dims[2] / sorted_dims[0] if sorted_dims[0] > 0 else 1
        features['aspect_ratio_2'] = sorted_dims[2] / sorted_dims[1] if sorted_dims[1] > 0 else 1
        features['aspect_ratio_3'] = sorted_dims[1] / sorted_dims[0] if sorted_dims[0] > 0 else 1
        
        # Mesh quality
        features['num_vertices'] = len(mesh.vertices)
        features['num_faces'] = len(mesh.faces)
        features['is_watertight'] = float(mesh.is_watertight)
        features['euler_number'] = mesh.euler_number
        
        return features
    
    @staticmethod
    def extract_curvature_features(mesh, sample_points=1000):
        """Extract curvature-based features"""
        features = {}
        
        try:
            # Sample points on surface
            points, face_idx = trimesh.sample.sample_surface(mesh, sample_points)
            
            # Get normals at sampled points
            normals = mesh.face_normals[face_idx]
            
            # Estimate curvature using normal variation
            # Higher variation in normals indicates higher curvature
            normal_variance = np.var(normals, axis=0)
            features['curvature_var_x'] = normal_variance[0]
            features['curvature_var_y'] = normal_variance[1]
            features['curvature_var_z'] = normal_variance[2]
            features['mean_curvature_var'] = np.mean(normal_variance)
            features['max_curvature_var'] = np.max(normal_variance)
            
            # Surface roughness (variation in point distances from centroid)
            centroid = points.mean(axis=0)
            distances = np.linalg.norm(points - centroid, axis=1)
            features['surface_roughness'] = np.std(distances)
            features['mean_radius'] = np.mean(distances)
            features['radius_variance'] = np.var(distances)
            
        except Exception as e:
            # Default values if extraction fails
            features['curvature_var_x'] = 0
            features['curvature_var_y'] = 0
            features['curvature_var_z'] = 0
            features['mean_curvature_var'] = 0
            features['max_curvature_var'] = 0
            features['surface_roughness'] = 0
            features['mean_radius'] = 0
            features['radius_variance'] = 0
            
        return features
    
    @staticmethod
    def extract_shape_descriptors(mesh, num_samples=2048):
        """Extract shape descriptors and moments"""
        features = {}
        
        try:
            # Sample points uniformly
            points, _ = trimesh.sample.sample_surface(mesh, num_samples)
            
            # Center the points
            centroid = points.mean(axis=0)
            centered_points = points - centroid
            
            # Principal Component Analysis for shape
            cov_matrix = np.cov(centered_points.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            # Shape descriptors from eigenvalues
            features['pca_1'] = eigenvalues[0]
            features['pca_2'] = eigenvalues[1]
            features['pca_3'] = eigenvalues[2]
            features['linearity'] = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0] if eigenvalues[0] > 0 else 0
            features['planarity'] = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0] if eigenvalues[0] > 0 else 0
            features['sphericity'] = eigenvalues[2] / eigenvalues[0] if eigenvalues[0] > 0 else 0
            
            # Statistical moments
            for i, axis in enumerate(['x', 'y', 'z']):
                axis_points = centered_points[:, i]
                features[f'skewness_{axis}'] = stats.skew(axis_points)
                features[f'kurtosis_{axis}'] = stats.kurtosis(axis_points)
                features[f'std_{axis}'] = np.std(axis_points)
            
            # Convex hull properties
            try:
                hull = mesh.convex_hull
                features['convexity'] = mesh.volume / hull.volume if hull.volume > 0 else 1
                features['hull_area_ratio'] = mesh.area / hull.area if hull.area > 0 else 1
            except:
                features['convexity'] = 1
                features['hull_area_ratio'] = 1
                
        except Exception as e:
            # Default values
            for key in ['pca_1', 'pca_2', 'pca_3', 'linearity', 'planarity', 'sphericity']:
                features[key] = 0
            for axis in ['x', 'y', 'z']:
                features[f'skewness_{axis}'] = 0
                features[f'kurtosis_{axis}'] = 0
                features[f'std_{axis}'] = 0
            features['convexity'] = 1
            features['hull_area_ratio'] = 1
            
        return features
    
    @staticmethod
    def extract_topological_features(mesh):
        """Extract topological features"""
        features = {}
        
        try:
            # Genus (number of holes)
            features['genus'] = mesh.body_count
            
            # Connected components
            components = mesh.split(only_watertight=False)
            features['num_components'] = len(components)
            
            # Largest component ratio
            if len(components) > 0:
                largest_volume = max(c.volume for c in components)
                features['largest_component_ratio'] = largest_volume / mesh.volume if mesh.volume > 0 else 1
            else:
                features['largest_component_ratio'] = 1
                
            # Edge statistics
            edges = mesh.edges_unique
            edge_lengths = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
            features['mean_edge_length'] = np.mean(edge_lengths)
            features['std_edge_length'] = np.std(edge_lengths)
            features['max_edge_length'] = np.max(edge_lengths)
            features['min_edge_length'] = np.min(edge_lengths)
            
        except Exception as e:
            features['genus'] = 0
            features['num_components'] = 1
            features['largest_component_ratio'] = 1
            features['mean_edge_length'] = 0
            features['std_edge_length'] = 0
            features['max_edge_length'] = 0
            features['min_edge_length'] = 0
            
        return features
    
    def extract_all_features(self, stl_path):
        """Extract all features from an STL file"""
        try:
            # Load mesh
            mesh = trimesh.load(stl_path)
            
            # Extract all feature groups
            features = {}
            features.update(self.extract_basic_features(mesh))
            features.update(self.extract_curvature_features(mesh))
            features.update(self.extract_shape_descriptors(mesh))
            features.update(self.extract_topological_features(mesh))
            
            return features
            
        except Exception as e:
            print(f"Error processing {stl_path}: {e}")
            return None

# ============== PART 2: DATA PREPARATION ==============

def prepare_traditional_ml_data(stl_dir='STL', labels_file='classification_labels_with_measurements.csv'):
    """Prepare data for traditional ML"""
    
    print("="*70)
    print("EXTRACTING GEOMETRIC FEATURES FROM STL FILES")
    print("="*70)
    
    # Load labels
    labels_df = pd.read_csv(labels_file)
    labels_df['filename'] = labels_df['filename'].str.replace('.npy', '').str.replace('.stl', '')
    
    # Initialize feature extractor
    extractor = GeometricFeatureExtractor()
    
    # Extract features for all STL files
    all_features = []
    all_labels = []
    all_measurements = []
    filenames = []
    
    measurement_cols = ['left_subclavian_diameter_mm', 'aortic_arch_diameter_mm', 'angle_degrees']
    
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Extracting features"):
        stl_file = row['filename'] + '.stl'
        stl_path = os.path.join(stl_dir, stl_file)
        
        if os.path.exists(stl_path):
            features = extractor.extract_all_features(stl_path)
            
            if features is not None:
                all_features.append(features)
                all_labels.append(row['label'])
                all_measurements.append(row[measurement_cols].values)
                filenames.append(row['filename'])
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Add anatomical measurements as features
    measurements_df = pd.DataFrame(all_measurements, columns=measurement_cols)
    
    # Combine all features
    combined_features = pd.concat([features_df, measurements_df], axis=1)
    
    print(f"\nExtracted {len(combined_features.columns)} features from {len(combined_features)} samples")
    print(f"Feature categories:")
    print(f"  - Geometric features: {len(features_df.columns)}")
    print(f"  - Anatomical measurements: {len(measurements_df.columns)}")
    
    # Handle any NaN or infinite values
    combined_features = combined_features.replace([np.inf, -np.inf], np.nan)
    combined_features = combined_features.fillna(0)
    
    return combined_features, np.array(all_labels), filenames

# ============== PART 3: TRADITIONAL ML MODELS ==============

class TraditionalMLClassifier:
    """Traditional ML models optimized for small datasets"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = 0
        
    def create_models(self):
        """Create various traditional ML models"""
        
        # Random Forest - good for small datasets
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            random_state=42
        )
        
        # XGBoost - often best performer
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            scale_pos_weight=1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Gradient Boosting
        self.models['gradient_boost'] = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
        
        # Support Vector Machine
        self.models['svm'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
        
        # Voting Classifier (Ensemble)
        self.models['ensemble'] = VotingClassifier(
            estimators=[
                ('rf', self.models['random_forest']),
                ('xgb', self.models['xgboost']),
                ('gb', self.models['gradient_boost'])
            ],
            voting='soft'
        )
    
    def train_and_evaluate(self, X, y, test_size=0.2, cv_folds=5):
        """Train and evaluate all models"""
        
        print("\n" + "="*70)
        print("TRAINING TRADITIONAL ML MODELS")
        print("="*70)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Number of features: {X_train.shape[1]}")
        
        # Create models
        self.create_models()
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{'-'*50}")
            print(f"Training {name.upper()}...")
            
            # Cross-validation on training set
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='accuracy')
            
            # Train on full training set
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # ROC AUC
            try:
                auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
            except:
                auc = 0
            
            results[name] = {
                'model': model,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': accuracy,
                'test_f1': f1,
                'test_auc': auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print(f"Test accuracy: {accuracy:.4f}")
            print(f"Test F1-score: {f1:.4f}")
            print(f"Test AUC: {auc:.4f}")
            
            # Update best model
            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model = name
        
        print(f"\n{'='*70}")
        print(f"BEST MODEL: {self.best_model.upper()}")
        print(f"Best Accuracy: {self.best_score:.4f}")
        print(f"{'='*70}")
        
        return results, X_test, y_test
    
    def plot_results(self, results, y_test):
        """Visualize results"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Traditional ML Models Performance', fontsize=16)
        
        # Plot 1: Model comparison
        model_names = list(results.keys())
        accuracies = [results[m]['test_accuracy'] for m in model_names]
        f1_scores = [results[m]['test_f1'] for m in model_names]
        
        ax = axes[0, 0]
        x = np.arange(len(model_names))
        width = 0.35
        ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        ax.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', '\n') for m in model_names], rotation=0)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Cross-validation scores
        ax = axes[0, 1]
        cv_means = [results[m]['cv_mean'] for m in model_names]
        cv_stds = [results[m]['cv_std'] for m in model_names]
        ax.errorbar(x, cv_means, yerr=cv_stds, marker='o', capsize=5, capthick=2)
        ax.set_xlabel('Models')
        ax.set_ylabel('CV Accuracy')
        ax.set_title('Cross-Validation Scores')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', '\n') for m in model_names], rotation=0)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Confusion Matrix for best model
        ax = axes[0, 2]
        best_model = self.best_model
        y_pred = results[best_model]['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix - {best_model.replace("_", " ").title()}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        # Plot 4: ROC Curves
        ax = axes[1, 0]
        for name, result in results.items():
            if result['y_pred_proba'] is not None:
                fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
                ax.plot(fpr, tpr, label=f"{name} (AUC={result['test_auc']:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Feature importance (for tree-based models)
        ax = axes[1, 1]
        if self.best_model in ['random_forest', 'xgboost', 'gradient_boost']:
            model = results[self.best_model]['model']
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:10]  # Top 10 features
                ax.barh(range(10), importances[indices])
                ax.set_yticks(range(10))
                ax.set_yticklabels([f"Feature {i}" for i in indices])
                ax.set_xlabel('Importance')
                ax.set_title(f'Top 10 Feature Importances - {self.best_model.replace("_", " ").title()}')
        
        # Plot 6: Prediction distribution
        ax = axes[1, 2]
        if results[self.best_model]['y_pred_proba'] is not None:
            probas = results[self.best_model]['y_pred_proba']
            ax.hist(probas[y_test == 0], bins=20, alpha=0.5, label='Class 0', color='blue')
            ax.hist(probas[y_test == 1], bins=20, alpha=0.5, label='Class 1', color='red')
            ax.set_xlabel('Predicted Probability (Class 1)')
            ax.set_ylabel('Count')
            ax.set_title('Prediction Probability Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('traditional_ml_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\n[SAVED] traditional_ml_results.png")
    
    def get_feature_importance(self, X, model_name=None):
        """Get feature importance for interpretation"""
        
        if model_name is None:
            model_name = self.best_model
            
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = X.columns if hasattr(X, 'columns') else [f'Feature_{i}' for i in range(X.shape[1])]
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 15 Most Important Features ({model_name}):")
            print("-" * 50)
            for idx, row in importance_df.head(15).iterrows():
                print(f"{row['feature']:30s}: {row['importance']:.4f}")
            
            return importance_df
        else:
            print(f"Model {model_name} doesn't support feature importance")
            return None

# ============== PART 4: MAIN EXECUTION ==============

def run_traditional_ml_classification():
    """Main function to run traditional ML classification"""
    
    print("\n" + "="*70)
    print("TRADITIONAL MACHINE LEARNING APPROACH")
    print("FOR SUBCLAVIAN ARTERY CLASSIFICATION")
    print("="*70)
    
    # Step 1: Extract features
    features, labels, filenames = prepare_traditional_ml_data()
    
    # Step 2: Train and evaluate models
    classifier = TraditionalMLClassifier()
    results, X_test, y_test = classifier.train_and_evaluate(features, labels, test_size=0.2, cv_folds=5)
    
    # Step 3: Visualize results
    classifier.plot_results(results, y_test)
    
    # Step 4: Get feature importance
    importance_df = classifier.get_feature_importance(features)
    
    # Step 5: Save results
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    for name, result in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Accuracy: {result['test_accuracy']:.4f}")
        print(f"  F1-Score: {result['test_f1']:.4f}")
        print(f"  AUC: {result['test_auc']:.4f}")
    
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {classifier.best_model.upper()}")
    print(f"BEST ACCURACY: {classifier.best_score:.4f}")
    print(f"{'='*70}")
    
    # Save model and scaler
    import joblib
    best_model = results[classifier.best_model]['model']
    joblib.dump(best_model, 'best_traditional_ml_model.pkl')
    joblib.dump(classifier.scaler, 'feature_scaler.pkl')
    print("\n[SAVED] best_traditional_ml_model.pkl")
    print("[SAVED] feature_scaler.pkl")
    
    return classifier, results

if __name__ == "__main__":
    classifier, results = run_traditional_ml_classification()