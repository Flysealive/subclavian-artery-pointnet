#!/usr/bin/env python3

import torch
import numpy as np
import sys
sys.path.append('./pointnet.pytorch')
from pointnet.model import PointNetCls
from subclavian_dataset import SubclavianDataset
import torch.nn.functional as F

def test_model(model_path, num_points=1024):
    """Test the trained PointNet model"""
    
    # Load the model
    classifier = PointNetCls(k=2, feature_transform=False)
    classifier.load_state_dict(torch.load(model_path, map_location='cpu'))
    classifier.eval()
    
    # Create test dataset
    test_dataset = SubclavianDataset(
        numpy_dir='numpy_arrays',
        csv_file='classification_labels.csv',
        split='test',
        npoints=num_points,
        data_augmentation=False
    )
    
    print(f"Testing on {len(test_dataset)} samples")
    
    correct = 0
    total = 0
    class_correct = [0, 0]
    class_total = [0, 0]
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            points, target = test_dataset[i]
            target = target[0]  # Remove extra dimension
            
            # Add batch dimension and transpose
            points = points.unsqueeze(0).transpose(2, 1)
            
            # Forward pass
            pred, _, _ = classifier(points)
            pred_choice = pred.data.max(1)[1][0]
            
            # Update statistics
            is_correct = pred_choice.eq(target.data).item()
            correct += is_correct
            total += 1
            
            class_total[target] += 1
            if is_correct:
                class_correct[target] += 1
            
            print(f"Sample {i+1}: Predicted={pred_choice.item()}, Actual={target.item()}, Correct={is_correct}")
    
    # Print results
    print(f"\nTest Results:")
    print(f"Overall Accuracy: {correct}/{total} = {correct/total:.4f}")
    
    for i in range(2):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            print(f"Class {i} Accuracy: {class_correct[i]}/{class_total[i]} = {acc:.4f}")
        else:
            print(f"Class {i}: No samples")
    
    return correct/total

if __name__ == '__main__':
    # Test the latest model
    model_path = 'cls/cls_model_9.pth'  # Last epoch
    accuracy = test_model(model_path)
    print(f"\nFinal test accuracy: {accuracy:.4f}")