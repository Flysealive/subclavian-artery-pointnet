import os
import numpy as np
import torch
from stl_to_voxel import stl_to_voxel
from voxel_cnn_model import VoxelCNN, VoxelResNet
from voxel_dataset import VoxelDataset

def test_voxel_conversion():
    """Test STL to voxel conversion"""
    print("=" * 50)
    print("Testing STL to Voxel Conversion")
    print("=" * 50)
    
    stl_files = [f for f in os.listdir('STL') if f.endswith('.stl')][:3]
    
    for stl_file in stl_files:
        stl_path = os.path.join('STL', stl_file)
        print(f"\nTesting: {stl_file}")
        
        try:
            voxel = stl_to_voxel(stl_path, voxel_size=64)
            print(f"  ✓ Voxel shape: {voxel.shape}")
            print(f"  ✓ Voxel range: [{voxel.min():.3f}, {voxel.max():.3f}]")
            print(f"  ✓ Non-zero voxels: {np.sum(voxel > 0)} / {voxel.size}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    return True

def test_models():
    """Test model architectures"""
    print("\n" + "=" * 50)
    print("Testing Model Architectures")
    print("=" * 50)
    
    batch_size = 2
    voxel_size = 64
    num_classes = 2
    
    dummy_input = torch.randn(batch_size, voxel_size, voxel_size, voxel_size)
    
    print("\n1. Testing VoxelCNN:")
    try:
        model = VoxelCNN(num_classes=num_classes, voxel_size=voxel_size)
        output = model(dummy_input)
        print(f"  ✓ Input shape: {dummy_input.shape}")
        print(f"  ✓ Output shape: {output.shape}")
        print(f"  ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n2. Testing VoxelResNet:")
    try:
        model = VoxelResNet(num_classes=num_classes, voxel_size=voxel_size)
        output = model(dummy_input)
        print(f"  ✓ Input shape: {dummy_input.shape}")
        print(f"  ✓ Output shape: {output.shape}")
        print(f"  ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    return True

def test_dataset():
    """Test dataset loading"""
    print("\n" + "=" * 50)
    print("Testing Dataset Loading")
    print("=" * 50)
    
    if not os.path.exists('voxel_arrays'):
        print("\nCreating sample voxel arrays...")
        os.makedirs('voxel_arrays', exist_ok=True)
        
        stl_files = [f for f in os.listdir('STL') if f.endswith('.stl')][:5]
        for stl_file in stl_files:
            stl_path = os.path.join('STL', stl_file)
            voxel = stl_to_voxel(stl_path, voxel_size=64)
            output_path = os.path.join('voxel_arrays', stl_file.replace('.stl', '.npy'))
            np.save(output_path, voxel)
            print(f"  Created: {output_path}")
    
    try:
        dataset = VoxelDataset(
            voxel_dir='voxel_arrays',
            labels_file='classification_labels.csv',
            mode='train',
            augment=False
        )
        
        if len(dataset) > 0:
            voxel, label = dataset[0]
            print(f"\n  ✓ Dataset size: {len(dataset)}")
            print(f"  ✓ Sample voxel shape: {voxel.shape}")
            print(f"  ✓ Sample label: {label}")
            print(f"  ✓ Label type: {type(label)}")
        else:
            print("  ! Dataset is empty")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    return True

def main():
    print("\n" + "=" * 70)
    print(" VOXEL-BASED 3D CNN PIPELINE TEST ")
    print("=" * 70)
    
    tests = [
        ("Voxel Conversion", test_voxel_conversion),
        ("Model Architectures", test_models),
        ("Dataset Loading", test_dataset)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} failed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 70)
    print(" TEST SUMMARY ")
    print("=" * 70)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 All tests passed! The voxel pipeline is ready to use.")
        print("\nTo train the model, run:")
        print("  python train_voxel_model.py --model cnn --epochs 100 --balanced")
        print("\nOr for ResNet architecture:")
        print("  python train_voxel_model.py --model resnet --epochs 100 --balanced")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()