"""
Cross-Validation Analysis for All Models
=========================================
Comprehensive k-fold cross-validation to get more reliable performance estimates
with small dataset (95 samples)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============== PART 1: CROSS-VALIDATION FOR TRADITIONAL ML ==============

def cross_validate_traditional_ml(n_folds=5):
    """
    Perform k-fold cross-validation for traditional ML models
    """
    print("\n" + "="*70)
    print("CROSS-VALIDATION: TRADITIONAL ML MODELS")
    print(f"Using {n_folds}-Fold Cross-Validation")
    print("="*70)
    
    # Import traditional ML components
    from traditional_ml_approach import GeometricFeatureExtractor, prepare_traditional_ml_data
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    import xgboost as xgb
    from sklearn.svm import SVC
    
    # Prepare data
    print("\nExtracting features...")
    features, labels, filenames = prepare_traditional_ml_data()
    
    # Initialize k-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Models to evaluate
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, class_weight='balanced', random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            use_label_encoder=False, eval_metric='logloss'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            min_samples_split=5, min_samples_leaf=2, subsample=0.8,
            random_state=42
        ),
        'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', 
                   class_weight='balanced', probability=True, random_state=42)
    }
    
    # Store results
    cv_results = {model_name: {'acc': [], 'f1': [], 'precision': [], 'recall': [], 'auc': []} 
                  for model_name in models}
    
    print(f"\nTotal samples: {len(features)}")
    print(f"Class distribution: {np.bincount(labels)}")
    print("\nStarting cross-validation...")
    print("-" * 50)
    
    # Perform cross-validation
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        print(f"\nFold {fold_idx + 1}/{n_folds}:")
        
        # Split data
        X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        print(f"  Train: {len(X_train)} samples, Val: {len(X_val)} samples")
        
        # Train and evaluate each model
        for model_name, model in models.items():
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_val_scaled)
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_val, y_pred, average='weighted')
            
            try:
                auc = roc_auc_score(y_val, y_pred_proba) if y_pred_proba is not None else 0
            except:
                auc = 0
            
            # Store results
            cv_results[model_name]['acc'].append(acc)
            cv_results[model_name]['f1'].append(f1)
            cv_results[model_name]['precision'].append(precision)
            cv_results[model_name]['recall'].append(recall)
            cv_results[model_name]['auc'].append(auc)
            
            print(f"  {model_name:20s}: Acc={acc:.4f}, F1={f1:.4f}")
    
    # Calculate statistics
    print("\n" + "="*70)
    print("CROSS-VALIDATION RESULTS - TRADITIONAL ML")
    print("="*70)
    
    results_summary = {}
    for model_name in models:
        results_summary[model_name] = {
            'acc_mean': np.mean(cv_results[model_name]['acc']),
            'acc_std': np.std(cv_results[model_name]['acc']),
            'f1_mean': np.mean(cv_results[model_name]['f1']),
            'f1_std': np.std(cv_results[model_name]['f1']),
            'precision_mean': np.mean(cv_results[model_name]['precision']),
            'recall_mean': np.mean(cv_results[model_name]['recall']),
            'auc_mean': np.mean(cv_results[model_name]['auc']),
            'all_accuracies': cv_results[model_name]['acc']
        }
        
        print(f"\n{model_name}:")
        print(f"  Accuracy: {results_summary[model_name]['acc_mean']:.4f} ± {results_summary[model_name]['acc_std']:.4f}")
        print(f"  F1-Score: {results_summary[model_name]['f1_mean']:.4f} ± {results_summary[model_name]['f1_std']:.4f}")
        print(f"  Precision: {results_summary[model_name]['precision_mean']:.4f}")
        print(f"  Recall: {results_summary[model_name]['recall_mean']:.4f}")
        print(f"  AUC: {results_summary[model_name]['auc_mean']:.4f}")
        print(f"  All fold accuracies: {[f'{acc:.4f}' for acc in cv_results[model_name]['acc']]}")
    
    return cv_results, results_summary

# ============== PART 2: CROSS-VALIDATION FOR DEEP LEARNING ==============

def cross_validate_hybrid_model(n_folds=5, epochs=30):
    """
    Perform k-fold cross-validation for hybrid deep learning model
    """
    print("\n" + "="*70)
    print("CROSS-VALIDATION: HYBRID DEEP LEARNING MODEL")
    print(f"Using {n_folds}-Fold Cross-Validation")
    print("="*70)
    
    # Import hybrid model components
    from hybrid_multimodal_model import HybridMultiModalNet, prepare_hybrid_data
    import pandas as pd
    import os
    
    # Prepare data if not exists
    if not os.path.exists('hybrid_data'):
        print("\nPreparing hybrid data...")
        prepare_hybrid_data()
    
    # Load all data
    labels_df = pd.read_csv('classification_labels_with_measurements.csv')
    labels_df['filename'] = labels_df['filename'].str.replace('.npy', '')
    
    # Filter for existing files
    pc_dir = 'hybrid_data/pointclouds'
    vox_dir = 'hybrid_data/voxels'
    
    valid_samples = []
    all_labels = []
    all_measurements = []
    
    measurement_cols = ['left_subclavian_diameter_mm', 'aortic_arch_diameter_mm', 'angle_degrees']
    
    for _, row in labels_df.iterrows():
        pc_file = os.path.join(pc_dir, row['filename'] + '.npy')
        vox_file = os.path.join(vox_dir, row['filename'] + '.npy')
        
        if os.path.exists(pc_file) and os.path.exists(vox_file):
            valid_samples.append(row['filename'])
            all_labels.append(row['label'])
            all_measurements.append(row[measurement_cols].values)
    
    valid_samples = np.array(valid_samples)
    all_labels = np.array(all_labels)
    all_measurements = np.array(all_measurements, dtype=np.float32)
    
    print(f"\nTotal samples: {len(valid_samples)}")
    print(f"Class distribution: {np.bincount(all_labels)}")
    
    # Normalize measurements
    scaler = StandardScaler()
    all_measurements = scaler.fit_transform(all_measurements)
    
    # Initialize k-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Store results
    cv_results = {'acc': [], 'f1': [], 'precision': [], 'recall': []}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("\nStarting cross-validation...")
    print("-" * 50)
    
    # Perform cross-validation
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(valid_samples, all_labels)):
        print(f"\nFold {fold_idx + 1}/{n_folds}:")
        print(f"  Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")
        
        # Split data
        train_samples = valid_samples[train_idx]
        val_samples = valid_samples[val_idx]
        train_labels = all_labels[train_idx]
        val_labels = all_labels[val_idx]
        train_measurements = all_measurements[train_idx]
        val_measurements = all_measurements[val_idx]
        
        # Create model
        model = HybridMultiModalNet(
            num_classes=2,
            num_points=2048,
            voxel_size=32,
            num_measurements=3
        ).to(device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_fold_acc = 0
        
        # Training loop (simplified)
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            
            # Mini-batch training
            batch_size = 8
            n_batches = len(train_samples) // batch_size
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(train_samples))
                
                batch_samples = train_samples[start_idx:end_idx]
                batch_labels = train_labels[start_idx:end_idx]
                batch_measurements = train_measurements[start_idx:end_idx]
                
                # Load batch data
                batch_pc = []
                batch_vox = []
                
                for sample_name in batch_samples:
                    pc = np.load(os.path.join(pc_dir, sample_name + '.npy'))
                    vox = np.load(os.path.join(vox_dir, sample_name + '.npy'))
                    batch_pc.append(pc)
                    batch_vox.append(vox)
                
                batch_pc = torch.FloatTensor(np.array(batch_pc)).to(device)
                batch_vox = torch.FloatTensor(np.array(batch_vox)).to(device)
                batch_measurements = torch.FloatTensor(batch_measurements).to(device)
                batch_labels = torch.LongTensor(batch_labels).to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch_pc, batch_vox, batch_measurements)
                loss = criterion(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == batch_labels).sum().item()
            
            train_acc = train_correct / len(train_samples)
            scheduler.step()
            
            # Validation
            if epoch % 5 == 0 or epoch == epochs - 1:
                model.eval()
                val_preds = []
                
                with torch.no_grad():
                    for i, sample_name in enumerate(val_samples):
                        pc = np.load(os.path.join(pc_dir, sample_name + '.npy'))
                        vox = np.load(os.path.join(vox_dir, sample_name + '.npy'))
                        
                        pc_tensor = torch.FloatTensor(pc).unsqueeze(0).to(device)
                        vox_tensor = torch.FloatTensor(vox).unsqueeze(0).to(device)
                        meas_tensor = torch.FloatTensor(val_measurements[i:i+1]).to(device)
                        
                        output = model(pc_tensor, vox_tensor, meas_tensor)
                        _, predicted = torch.max(output.data, 1)
                        val_preds.append(predicted.cpu().numpy()[0])
                
                val_acc = accuracy_score(val_labels, val_preds)
                val_f1 = f1_score(val_labels, val_preds, average='weighted')
                
                if val_acc > best_fold_acc:
                    best_fold_acc = val_acc
                
                if epoch % 10 == 0:
                    print(f"    Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        # Final evaluation for this fold
        model.eval()
        val_preds = []
        
        with torch.no_grad():
            for i, sample_name in enumerate(val_samples):
                pc = np.load(os.path.join(pc_dir, sample_name + '.npy'))
                vox = np.load(os.path.join(vox_dir, sample_name + '.npy'))
                
                pc_tensor = torch.FloatTensor(pc).unsqueeze(0).to(device)
                vox_tensor = torch.FloatTensor(vox).unsqueeze(0).to(device)
                meas_tensor = torch.FloatTensor(val_measurements[i:i+1]).to(device)
                
                output = model(pc_tensor, vox_tensor, meas_tensor)
                _, predicted = torch.max(output.data, 1)
                val_preds.append(predicted.cpu().numpy()[0])
        
        # Calculate metrics
        acc = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds, average='weighted')
        precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
        recall = recall_score(val_labels, val_preds, average='weighted')
        
        cv_results['acc'].append(acc)
        cv_results['f1'].append(f1)
        cv_results['precision'].append(precision)
        cv_results['recall'].append(recall)
        
        print(f"  Final: Acc={acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    
    # Calculate statistics
    print("\n" + "="*70)
    print("CROSS-VALIDATION RESULTS - HYBRID DEEP LEARNING")
    print("="*70)
    
    results_summary = {
        'acc_mean': np.mean(cv_results['acc']),
        'acc_std': np.std(cv_results['acc']),
        'f1_mean': np.mean(cv_results['f1']),
        'f1_std': np.std(cv_results['f1']),
        'precision_mean': np.mean(cv_results['precision']),
        'recall_mean': np.mean(cv_results['recall']),
        'all_accuracies': cv_results['acc']
    }
    
    print(f"\nHybrid Multi-Modal Model:")
    print(f"  Accuracy: {results_summary['acc_mean']:.4f} ± {results_summary['acc_std']:.4f}")
    print(f"  F1-Score: {results_summary['f1_mean']:.4f} ± {results_summary['f1_std']:.4f}")
    print(f"  Precision: {results_summary['precision_mean']:.4f}")
    print(f"  Recall: {results_summary['recall_mean']:.4f}")
    print(f"  All fold accuracies: {[f'{acc:.4f}' for acc in cv_results['acc']]}")
    
    return cv_results, results_summary

# ============== PART 3: COMPARISON AND VISUALIZATION ==============

def compare_all_models():
    """
    Run cross-validation for all models and compare results
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE CROSS-VALIDATION COMPARISON")
    print("="*70)
    
    # Run traditional ML cross-validation
    trad_cv_results, trad_summary = cross_validate_traditional_ml(n_folds=5)
    
    # Run hybrid model cross-validation
    hybrid_cv_results, hybrid_summary = cross_validate_hybrid_model(n_folds=5, epochs=30)
    
    # Combine results for comparison
    all_results = {}
    for model_name, summary in trad_summary.items():
        all_results[model_name] = summary
    all_results['Hybrid Deep Learning'] = hybrid_summary
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Cross-Validation Results: Traditional ML vs Deep Learning', fontsize=16)
    
    # Plot 1: Accuracy comparison
    ax = axes[0, 0]
    model_names = list(all_results.keys())
    acc_means = [all_results[m]['acc_mean'] for m in model_names]
    acc_stds = [all_results[m]['acc_std'] for m in model_names]
    
    x = np.arange(len(model_names))
    bars = ax.bar(x, acc_means, yerr=acc_stds, capsize=5, alpha=0.7)
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy')
    ax.set_title('Cross-Validation Accuracy (Mean ± Std)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ', '\n') for m in model_names], rotation=0, fontsize=9)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(acc_means, acc_stds)):
        ax.text(i, mean + std + 0.01, f'{mean:.3f}', ha='center', fontsize=9)
    
    # Plot 2: F1-Score comparison
    ax = axes[0, 1]
    f1_means = [all_results[m]['f1_mean'] for m in model_names]
    f1_stds = [all_results[m]['f1_std'] for m in model_names]
    
    bars = ax.bar(x, f1_means, yerr=f1_stds, capsize=5, alpha=0.7, color='orange')
    ax.set_xlabel('Models')
    ax.set_ylabel('F1-Score')
    ax.set_title('Cross-Validation F1-Score (Mean ± Std)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ', '\n') for m in model_names], rotation=0, fontsize=9)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Box plot of accuracies across folds
    ax = axes[0, 2]
    all_accs = [all_results[m]['all_accuracies'] for m in model_names]
    bp = ax.boxplot(all_accs, labels=[m.replace(' ', '\n') for m in model_names])
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Distribution Across Folds')
    ax.grid(True, alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=9)
    
    # Plot 4: Precision vs Recall
    ax = axes[1, 0]
    precisions = [all_results[m]['precision_mean'] for m in model_names]
    recalls = [all_results[m]['recall_mean'] for m in model_names]
    
    ax.scatter(recalls, precisions, s=100, alpha=0.7)
    for i, model in enumerate(model_names):
        ax.annotate(model, (recalls[i], precisions[i]), fontsize=8,
                   xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Recall')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.5, 1])
    ax.set_ylim([0.5, 1])
    
    # Plot 5: Stability analysis
    ax = axes[1, 1]
    stabilities = [(1 - all_results[m]['acc_std']) for m in model_names]
    bars = ax.bar(x, stabilities, alpha=0.7, color='green')
    ax.set_xlabel('Models')
    ax.set_ylabel('Stability Score (1 - Std)')
    ax.set_title('Model Stability Across Folds')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' ', '\n') for m in model_names], rotation=0, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Final ranking
    ax = axes[1, 2]
    # Calculate overall score (weighted combination)
    overall_scores = []
    for m in model_names:
        score = (all_results[m]['acc_mean'] * 0.4 + 
                all_results[m]['f1_mean'] * 0.3 + 
                all_results[m]['precision_mean'] * 0.15 + 
                all_results[m]['recall_mean'] * 0.15)
        overall_scores.append(score)
    
    # Sort by score
    sorted_idx = np.argsort(overall_scores)[::-1]
    sorted_models = [model_names[i] for i in sorted_idx]
    sorted_scores = [overall_scores[i] for i in sorted_idx]
    
    y_pos = np.arange(len(sorted_models))
    bars = ax.barh(y_pos, sorted_scores, alpha=0.7, color='purple')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_models, fontsize=9)
    ax.set_xlabel('Overall Score')
    ax.set_title('Final Model Ranking')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, score in enumerate(sorted_scores):
        ax.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('cross_validation_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n[SAVED] cross_validation_comparison.png")
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL CROSS-VALIDATION SUMMARY")
    print("="*70)
    
    # Sort by accuracy
    sorted_by_acc = sorted(all_results.items(), key=lambda x: x[1]['acc_mean'], reverse=True)
    
    print("\nModels Ranked by Cross-Validation Accuracy:")
    print("-" * 50)
    for rank, (model_name, summary) in enumerate(sorted_by_acc, 1):
        print(f"{rank}. {model_name:25s}: {summary['acc_mean']:.4f} ± {summary['acc_std']:.4f}")
    
    # Statistical significance note
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("\n1. With 95 samples and 5-fold CV, each fold has ~19 test samples")
    print("2. Small test sets lead to high variance in results")
    print("3. Differences < 5% are likely not statistically significant")
    print("4. More data is needed for reliable model selection")
    
    return all_results

if __name__ == "__main__":
    # Run comprehensive comparison
    results = compare_all_models()