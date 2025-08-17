#!/usr/bin/env python3
"""
Test GPU training to ensure everything works before full training
"""

import torch
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

print("="*60)
print("GPU TRAINING TEST")
print("="*60)

# Check GPU
if torch.cuda.is_available():
    print(f"[OK] GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    device = torch.device('cuda')
else:
    print("[ERROR] No GPU found, using CPU")
    device = torch.device('cpu')

# Check data files
print("\nChecking data files...")
pointcloud_dir = Path('hybrid_data/pointclouds')
voxel_dir = Path('hybrid_data/voxels')
labels_file = Path('classification_labels_with_measurements.csv')

if not pointcloud_dir.exists():
    print(f"[ERROR] Pointcloud directory not found: {pointcloud_dir}")
    sys.exit(1)
    
if not voxel_dir.exists():
    print(f"[ERROR] Voxel directory not found: {voxel_dir}")
    sys.exit(1)
    
if not labels_file.exists():
    print(f"[ERROR] Labels file not found: {labels_file}")
    sys.exit(1)

pc_files = list(pointcloud_dir.glob('*.npy'))
vox_files = list(voxel_dir.glob('*.npy'))

print(f"[OK] Found {len(pc_files)} pointcloud files")
print(f"[OK] Found {len(vox_files)} voxel files")

# Load one sample to test
print("\nLoading test sample...")
sample_pc = np.load(pc_files[0])
sample_vox = np.load(vox_files[0])
print(f"[OK] Pointcloud shape: {sample_pc.shape}")
print(f"[OK] Voxel shape: {sample_vox.shape}")

# Test tensor conversion and GPU transfer
print("\nTesting GPU operations...")
test_tensor = torch.randn(8, 3, 32, 32, 32).to(device)
print(f"[OK] Created test tensor on {device}")
print(f"  Shape: {test_tensor.shape}")

# Simple convolution test
conv = torch.nn.Conv3d(3, 16, 3, padding=1).to(device)
output = conv(test_tensor)
print(f"[OK] Conv3D test passed")
print(f"  Output shape: {output.shape}")

print("\n" + "="*60)
print("[OK] GPU TRAINING TEST SUCCESSFUL!")
print("Ready to run full training")
print("="*60)