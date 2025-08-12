"""
Simple GPU Training Script for Voxel CNN
=========================================
This script automatically uses your NVIDIA GPU for fast training
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("VOXEL CNN GPU TRAINING")
    print("="*60)
    
    # Step 1: Check if PyTorch is installed and can access GPU
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__} installed")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[OK] GPU detected: {gpu_name}")
            print(f"[OK] GPU Memory: {gpu_memory:.1f}GB")
            device = 'cuda'
        else:
            print("[WARNING] No GPU detected, using CPU (will be slower)")
            device = 'cpu'
    except ImportError:
        print("[ERROR] PyTorch not installed!")
        print("Please run: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return
    
    # Step 2: Check if other dependencies are installed
    try:
        import numpy
        import pandas
        import trimesh
        from tqdm import tqdm
        print("[OK] All dependencies installed")
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Please run: pip install numpy pandas trimesh tqdm scikit-learn matplotlib seaborn")
        return
    
    # Step 3: Convert STL to voxels if needed
    if not os.path.exists('voxel_arrays'):
        print("\n[INFO] Converting STL files to voxels...")
        print("This is a one-time process...")
        
        try:
            from stl_to_voxel import convert_dataset_to_voxels
            convert_dataset_to_voxels(voxel_size=64)
            print("[OK] Voxel conversion complete!")
        except Exception as e:
            print(f"[ERROR] Failed to convert STL files: {e}")
            return
    else:
        voxel_count = len([f for f in os.listdir('voxel_arrays') if f.endswith('.npy')])
        print(f"[OK] Found {voxel_count} voxel files")
    
    # Step 4: Load the model and data
    print("\n[INFO] Setting up model and data...")
    
    try:
        from voxel_cnn_model import VoxelCNN
        from voxel_dataset import get_voxel_dataloaders
        
        # Create data loaders with GPU optimization
        batch_size = 16 if device == 'cuda' else 8
        num_workers = 4 if device == 'cuda' else 0
        
        train_loader, test_loader = get_voxel_dataloaders(
            batch_size=batch_size,
            balanced=True,
            num_workers=num_workers
        )
        print(f"[OK] Data loaded (batch_size={batch_size})")
        
        # Create model
        model = VoxelCNN(num_classes=2, voxel_size=64)
        model = model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[OK] Model created ({total_params:,} parameters)")
        
    except Exception as e:
        print(f"[ERROR] Failed to load model/data: {e}")
        return
    
    # Step 5: Start training
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    try:
        from gpu_voxel_training import GPUVoxelTrainer
        
        trainer = GPUVoxelTrainer(model, device, "voxel_cnn", use_amp=(device=='cuda'))
        
        print("[INFO] Training for 100 epochs with early stopping...")
        print("[INFO] This will take approximately 20-40 minutes on GPU")
        print("[INFO] Press Ctrl+C to stop early\n")
        
        history = trainer.train(
            train_loader, test_loader,
            epochs=100,
            lr=0.001,
            patience=20
        )
        
        print("\n[OK] Training complete!")
        print(f"[OK] Best accuracy: {max(history['val_acc']):.2%}")
        
        # Save the model
        torch.save(model.state_dict(), 'trained_voxel_model.pth')
        print("[OK] Model saved as 'trained_voxel_model.pth'")
        
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
        torch.save(model.state_dict(), 'interrupted_voxel_model.pth')
        print("[OK] Partial model saved as 'interrupted_voxel_model.pth'")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        
        # Try simpler training if GPU training fails
        print("\n[INFO] Attempting simple training...")
        
        import torch.nn as nn
        import torch.optim as optim
        from tqdm import tqdm
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print("[INFO] Running 10 quick epochs...")
        
        for epoch in range(10):
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
                data, labels = data.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            accuracy = correct / total
            print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Acc={accuracy:.2%}")
        
        torch.save(model.state_dict(), 'simple_voxel_model.pth')
        print("[OK] Simple model saved as 'simple_voxel_model.pth'")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        print("\nFor help, check:")
        print("1. Is your NVIDIA GPU driver installed?")
        print("2. Run: python check_gpu.py")
        print("3. Install packages: pip install -r voxel_requirements.txt")