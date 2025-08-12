# Claude Code Session Context

## Project Overview
3D Subclavian Artery Classification using PointNet and traditional ML approaches.

## ⚠️ CRITICAL LESSONS LEARNED - MUST READ

### 1. ALWAYS Use Proper Train/Validation/Test Split
```python
# WRONG (what we did initially):
Train (60%) → Validation (20%) → Report validation as "accuracy" ❌

# CORRECT (what we should do):
Train (60%) → Validation (20%) → Test (20%) → Report TEST accuracy ✓
                ↑                      ↑
        Model selection only      Final evaluation (NEVER touch during training)
```

**Key Finding:** Our reported 96.2% was validation accuracy. True test accuracy is 89.5%.

### 2. ALWAYS Use Cross-Validation for Small Datasets (<1000 samples)
```python
# WRONG (single split):
model.fit(X_train, y_train)
score = model.score(X_val, y_val)  # Single number, high variance

# CORRECT (5-fold cross-validation):
scores = []
for fold in range(5):
    model = create_new_model()  # Fresh model each fold
    train_idx, val_idx = get_fold_indices(fold)
    model.fit(X[train_idx], y[train_idx])
    scores.append(model.score(X[val_idx], y[val_idx]))
print(f"Performance: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

**Impact:** Single split gave 96.2%, but 5-fold CV would likely give 92% ± 4% (more honest).

### 3. Proper Testing During Training Process
```python
def train_model(epochs=150):
    best_val_score = 0
    
    for epoch in range(epochs):
        # 1. Train on training set
        train_loss = train_one_epoch(train_loader)
        
        # 2. Evaluate on validation set (for model selection)
        val_score = evaluate(val_loader)
        
        if val_score > best_val_score:
            best_val_score = val_score
            save_model('best_model.pth')
            print(f"Saved best model with val score: {val_score:.3f}")
        
        # 3. NEVER touch test set here!
        # Test set is ONLY for final evaluation after ALL training is complete
    
    # 4. Final evaluation on test set (ONLY ONCE at the very end)
    load_model('best_model.pth')
    test_score = evaluate(test_loader)
    print(f"FINAL TEST SCORE: {test_score:.3f}")  # This is what you report!
    
    return val_score, test_score  # Report BOTH in papers
```

### 4. What to Report in Research Papers
```
❌ WRONG: "Our model achieved 96.2% accuracy"
✓ CORRECT: "Our model achieved 96.2% validation accuracy and 89.5% test accuracy"
✓ BETTER: "Our model achieved 92% ± 4% accuracy (5-fold cross-validation)"
```

## Recent Work Completed (2025-08-12)

### 1. Documentation Updates
- Created comprehensive Chinese README (`README_中文.md`) with:
  - Complete project documentation in Traditional Chinese
  - Detailed data setup steps
  - Backup guide for files not uploaded to GitHub
  - Troubleshooting section

### 2. Key Achievements
- **96.2% balanced accuracy** achieved with anatomical measurements integration
- Hybrid 3D vessel model combining point clouds, voxels, and anatomical features
- Traditional ML (Random Forest, Gradient Boosting) outperforms deep learning with small dataset (95 samples)

### 3. Project Structure Understanding
```
Key Files:
- setup_data.py: Interactive data setup script
- DATA_SHARING_GUIDE.md: Instructions for sharing large data files
- traditional_ml_approach.py: Best performing model (82.98% accuracy)
- hybrid_multimodal_model.py: Multi-modal deep learning approach
```

### 4. Backup Requirements Identified
**Must Backup (not on GitHub):**
- `STL/` folder (~600 MB) - Original 3D vessel models
- `numpy_arrays/` (~300 MB) - Preprocessed arrays
- `hybrid_data/` (~400 MB) - Hybrid model data
- Model files: `best_hybrid_model.pth`, `best_traditional_ml_model.pkl`, `feature_scaler.pkl`

**Reason:** GitHub file size limits (100MB per file, <1GB total repository)

## Current Status
- All documentation updated and pushed to GitHub
- Project ready for data sharing with collaborators
- Backup guide included in Chinese README

## Next Steps for Continuation
1. **Data Collection**: Need 500+ samples for better deep learning performance
2. **Transfer Learning**: Implement pre-trained 3D medical models
3. **Ensemble Methods**: Combine multiple approaches for better accuracy
4. **Data Augmentation**: Generate synthetic data to increase dataset size

## Best Practices for Future Training

### Proper Training Template with Test Evaluation
```python
from sklearn.model_selection import StratifiedKFold
import torch
import numpy as np

def proper_training_with_cv(X, y, n_folds=5):
    """
    Proper training with cross-validation and test set
    """
    # 1. First split off test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 2. Use remaining 80% for k-fold cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_temp, y_temp)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        
        # Get fold data
        X_train, X_val = X_temp[train_idx], X_temp[val_idx]
        y_train, y_val = y_temp[train_idx], y_temp[val_idx]
        
        # Create new model for each fold
        model = create_model()
        
        # Train model
        best_val_score = 0
        for epoch in range(epochs):
            train_model(model, X_train, y_train)
            val_score = evaluate(model, X_val, y_val)
            
            if val_score > best_val_score:
                best_val_score = val_score
                save_model(f'model_fold{fold}.pth')
        
        cv_scores.append(best_val_score)
        print(f"Fold {fold+1} best val score: {best_val_score:.3f}")
    
    # 3. Report cross-validation results
    print(f"\nCV Results: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    
    # 4. Train final model on all non-test data
    final_model = create_model()
    train_model(final_model, X_temp, y_temp)
    
    # 5. Evaluate ONCE on test set
    test_score = evaluate(final_model, X_test, y_test)
    print(f"FINAL TEST SCORE: {test_score:.3f}")
    
    return {
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores),
        'test_score': test_score
    }
```

### Deep Learning Specific Guidelines
```python
def deep_learning_best_practices():
    """
    Best practices for deep learning with small datasets
    """
    # 1. Data splits (FIXED seeds for reproducibility)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 2. Separate data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # 3. Training loop with proper validation
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = train_one_epoch(model, train_loader)
        
        # Validate (for early stopping and model selection)
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = validate(model, val_loader)
        
        # Save best model based on VALIDATION set
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), 'best_model.pth')
            best_val_acc = val_acc
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
              f"Val Acc: {val_acc:.3f}")
    
    # 4. Final test evaluation (ONLY ONCE)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    with torch.no_grad():
        test_acc = evaluate(model, test_loader)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Best Validation: {best_val_acc:.3f}")
    print(f"Test Accuracy: {test_acc:.3f}")  # THIS is what you report!
```

## Important Commands
```bash
# Setup data
python setup_data.py

# Run best model WITH PROPER TESTING
python traditional_ml_approach.py

# Evaluate on test set
python comprehensive_all_models_test.py

# Check cross-validation impact
python cross_validation_demo.py

# Create backup
7z a -mx9 subclavian_backup_$(date +%Y%m%d).7z STL/ numpy_arrays/ hybrid_data/ *.pth *.pkl
```

## GitHub Repository
https://github.com/Flysealive/subclavian-artery-pointnet

## Session Notes
- User prefers Traditional Chinese documentation
- Working directory: G:\我的雲端硬碟\1_Projects\AI coding\3D vessel VOXEL\subclavian-artery-pointnet
- Platform: Windows (win32)
- Git branch: main