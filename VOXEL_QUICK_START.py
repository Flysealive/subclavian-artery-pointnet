"""
VOXEL-BASED 3D CNN FOR SUBCLAVIAN ARTERY CLASSIFICATION
========================================================

This is a complete voxel-based approach for 3D vessel classification,
replacing the point cloud method with voxel representation.

QUICK START GUIDE:
-----------------

1. Install Requirements:
   pip install -r voxel_requirements.txt

2. Convert STL files to Voxels:
   python stl_to_voxel.py
   
   This will create voxel arrays in 'voxel_arrays/' directory

3. Train the Model:
   
   Option A - Basic CNN (faster, good for testing):
   python train_voxel_model.py --model cnn --epochs 100 --batch_size 8 --balanced
   
   Option B - ResNet (better accuracy, slower):
   python train_voxel_model.py --model resnet --epochs 200 --batch_size 4 --balanced
   
   Additional options:
   --voxel_size 64    # Size of voxel grid (32, 64, or 128)
   --lr 0.001         # Learning rate
   --patience 20      # Early stopping patience

4. Test the Pipeline:
   python test_voxel_pipeline.py

FILES OVERVIEW:
--------------
- stl_to_voxel.py: Converts STL meshes to voxel grids
- voxel_cnn_model.py: Contains VoxelCNN and VoxelResNet architectures
- voxel_dataset.py: Dataset loader with augmentation
- train_voxel_model.py: Main training script
- test_voxel_pipeline.py: Tests all components

KEY DIFFERENCES FROM POINT CLOUD:
---------------------------------
1. Voxel representation preserves spatial structure better
2. Uses 3D convolutions instead of 1D operations
3. Can capture volume and density information
4. More memory intensive but often more accurate
5. Better for capturing global shape features

VOXEL SIZES:
-----------
- 32x32x32: Fast, low memory, less detail
- 64x64x64: Good balance (recommended)
- 128x128x128: High detail, slow, high memory

EXPECTED PERFORMANCE:
--------------------
With balanced dataset and proper training:
- Accuracy: 85-95%
- Training time: ~30-60 min for 100 epochs on GPU
- Memory usage: ~4-8 GB GPU RAM for batch_size=8

"""

import sys
import os

def check_environment():
    print("Checking environment...")
    
    try:
        import numpy
        print("✓ NumPy installed")
    except:
        print("✗ NumPy not installed - run: pip install numpy")
    
    try:
        import trimesh
        print("✓ Trimesh installed")
    except:
        print("✗ Trimesh not installed - run: pip install trimesh")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ PyTorch installed (GPU available: {torch.cuda.get_device_name(0)})")
        else:
            print("✓ PyTorch installed (CPU only)")
    except:
        print("✗ PyTorch not installed - run: pip install torch torchvision")
    
    if os.path.exists('STL'):
        stl_count = len([f for f in os.listdir('STL') if f.endswith('.stl')])
        print(f"✓ Found {stl_count} STL files")
    else:
        print("✗ STL directory not found")
    
    if os.path.exists('classification_labels.csv'):
        print("✓ Labels file found")
    else:
        print("✗ classification_labels.csv not found")
    
    print("\nTo start training, run:")
    print("python train_voxel_model.py --model cnn --epochs 100 --balanced")

if __name__ == "__main__":
    check_environment()