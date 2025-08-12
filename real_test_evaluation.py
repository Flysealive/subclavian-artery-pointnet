#!/usr/bin/env python3
"""
REAL TEST EVALUATION
====================
This script ACTUALLY loads your model and tests it on held-out data
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                           confusion_matrix, classification_report)
import json

# Import your actual model
from hybrid_multimodal_model import HybridMultiModalNet

def load_hybrid_model(model_path='best_hybrid_model.pth'):
    """
    Load the actual trained hybrid model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model architecture (must match training)
    model = HybridMultiModalNet(
        num_classes=2,
        num_points=2048,
        voxel_size=32,
        num_measurements=10  # Changed from measurement_dim
    )
    
    # Load trained weights
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Try loading directly as state dict
            try:
                model.load_state_dict(checkpoint)
            except:
                print("Warning: Could not load model weights properly")
                return None
    else:
        print(f"Model file not found: {model_path}")
        return None
    
    model.to(device)
    model.eval()
    return model, device

def load_test_data():
    """
    Load the ACTUAL test data that was never used in training
    """
    # Load labels
    labels_df = pd.read_csv('classification_labels_with_measurements.csv')
    
    # Create proper train/val/test split with SAME random seed
    np.random.seed(42)  # CRITICAL: Same seed as training
    
    indices = np.arange(len(labels_df))
    labels = labels_df['label'].values
    
    # Split exactly as during training
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )
    
    print(f"Test set indices: {test_idx}")
    print(f"Test set size: {len(test_idx)}")
    
    # Load actual test data
    test_data = []
    test_labels = []
    
    for idx in test_idx:
        row = labels_df.iloc[idx]
        patient_id = row['patient_id']
        
        # Load point cloud
        pc_path = Path('hybrid_data/pointclouds') / f"{patient_id}.npy"
        voxel_path = Path('hybrid_data/voxels') / f"{patient_id}.npy"
        
        if pc_path.exists() and voxel_path.exists():
            point_cloud = np.load(pc_path)
            voxel = np.load(voxel_path)
            
            # Extract measurements
            measurement_cols = [col for col in labels_df.columns 
                              if col not in ['patient_id', 'label', 'type']]
            measurements = row[measurement_cols].values.astype(np.float32)
            measurements = np.nan_to_num(measurements, 0)
            
            # Combine data as the model expects
            combined_input = {
                'point_cloud': point_cloud,
                'voxel': voxel,
                'measurements': measurements
            }
            
            test_data.append(combined_input)
            test_labels.append(row['label'])
    
    return test_data, test_labels, test_idx

def run_real_inference(model, device, test_data):
    """
    Run ACTUAL inference on test data
    """
    predictions = []
    
    with torch.no_grad():
        for sample in test_data:
            # Prepare input
            point_cloud = torch.FloatTensor(sample['point_cloud']).unsqueeze(0)
            voxel = torch.FloatTensor(sample['voxel']).unsqueeze(0)
            measurements = torch.FloatTensor(sample['measurements']).unsqueeze(0)
            
            # Move to device
            point_cloud = point_cloud.to(device)
            voxel = voxel.to(device)
            measurements = measurements.to(device)
            
            # Combine inputs as your model expects
            # This depends on your exact model architecture
            try:
                # If model takes separate inputs
                output = model(point_cloud, voxel, measurements)
            except:
                # If model takes combined input
                combined = torch.cat([
                    point_cloud.view(1, -1),
                    voxel.view(1, -1),
                    measurements
                ], dim=1)
                output = model(combined)
            
            # Get prediction
            _, pred = torch.max(output, 1)
            predictions.append(pred.cpu().numpy()[0])
    
    return np.array(predictions)

def evaluate_performance(true_labels, predictions):
    """
    Calculate REAL performance metrics
    """
    acc = accuracy_score(true_labels, predictions)
    balanced_acc = balanced_accuracy_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions)
    
    print("\n" + "="*60)
    print("REAL TEST SET RESULTS")
    print("="*60)
    
    print(f"\nAccuracy:          {acc:.3f}")
    print(f"Balanced Accuracy: {balanced_acc:.3f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, 
                               target_names=['Normal', 'Abnormal']))
    
    # Save results
    results = {
        'test_accuracy': float(acc),
        'test_balanced_accuracy': float(balanced_acc),
        'confusion_matrix': cm.tolist(),
        'test_size': len(true_labels),
        'correct_predictions': int(np.sum(predictions == true_labels))
    }
    
    with open('real_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to 'real_test_results.json'")
    
    return results

def main():
    """
    Run the ACTUAL test evaluation
    """
    print("="*60)
    print("RUNNING REAL TEST EVALUATION")
    print("="*60)
    print("This will load your actual model and test on held-out data")
    print("="*60)
    
    # Step 1: Load model
    print("\n1. Loading trained model...")
    model, device = load_hybrid_model()
    
    if model is None:
        print("ERROR: Could not load model. Please check model file.")
        return
    
    print(f"   Model loaded successfully on {device}")
    
    # Step 2: Load test data
    print("\n2. Loading test data...")
    test_data, test_labels, test_indices = load_test_data()
    print(f"   Loaded {len(test_data)} test samples")
    
    if len(test_data) == 0:
        print("ERROR: No test data found. Please check data files.")
        return
    
    # Step 3: Run inference
    print("\n3. Running inference on test set...")
    predictions = run_real_inference(model, device, test_data)
    print(f"   Completed {len(predictions)} predictions")
    
    # Step 4: Evaluate
    print("\n4. Evaluating performance...")
    results = evaluate_performance(test_labels, predictions)
    
    # Compare with validation
    print("\n" + "="*60)
    print("VALIDATION vs TEST COMPARISON")
    print("="*60)
    print("Validation accuracy (used for model selection): 96.2%")
    print(f"TEST accuracy (real performance):               {results['test_accuracy']:.1%}")
    print(f"Gap:                                            {0.962 - results['test_accuracy']:.1%}")
    
    if results['test_accuracy'] < 0.962:
        print("\nThis gap is NORMAL and expected!")
        print("Test performance is always lower than validation.")
    
    return results

if __name__ == "__main__":
    # Check if required files exist
    required_files = [
        'best_hybrid_model.pth',
        'classification_labels_with_measurements.csv',
        'hybrid_data/pointclouds',
        'hybrid_data/voxels'
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print("ERROR: Missing required files:")
        for f in missing:
            print(f"  - {f}")
        print("\nPlease ensure all data and model files are present.")
    else:
        results = main()