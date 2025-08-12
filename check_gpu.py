import sys
import subprocess

def check_pytorch_gpu():
    try:
        import torch
        print("[OK] PyTorch is installed")
        print(f"  Version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print("\n[OK] CUDA is available!")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            print(f"  Current GPU: {torch.cuda.current_device()}")
            print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
            
            # Get GPU properties
            props = torch.cuda.get_device_properties(0)
            print(f"\n  GPU Properties:")
            print(f"    - Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"    - Multiprocessors: {props.multi_processor_count}")
            print(f"    - CUDA Capability: {props.major}.{props.minor}")
            
            # Test GPU computation
            print("\n  Testing GPU computation...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print(f"  [OK] GPU computation successful!")
            print(f"  [OK] Result shape: {z.shape}")
            
            return True
        else:
            print("\n[ERROR] CUDA is NOT available")
            print("  PyTorch is installed but cannot access GPU")
            return False
            
    except ImportError:
        print("[ERROR] PyTorch is not installed")
        return False

def install_pytorch_cuda():
    print("\nWould you like to install PyTorch with CUDA support?")
    print("This will install PyTorch with CUDA 12.1 support")
    
    response = input("Install? (y/n): ").lower()
    if response == 'y':
        print("\nInstalling PyTorch with CUDA 12.1...")
        cmd = [sys.executable, "-m", "pip", "install", 
               "torch", "torchvision", "torchaudio", 
               "--index-url", "https://download.pytorch.org/whl/cu121"]
        
        try:
            subprocess.run(cmd, check=True)
            print("\n[OK] Installation complete!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Installation failed: {e}")
            return False
    return False

if __name__ == "__main__":
    print("="*60)
    print("GPU CHECK FOR VOXEL TRAINING")
    print("="*60)
    
    has_gpu = check_pytorch_gpu()
    
    if not has_gpu:
        print("\n" + "="*60)
        print("INSTALLATION REQUIRED")
        print("="*60)
        
        try:
            import torch
            # PyTorch installed but no CUDA
            print("\nPyTorch is installed but CUDA support is missing.")
            print("You need to reinstall PyTorch with CUDA support.")
            print("\nRun this command:")
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        except ImportError:
            # PyTorch not installed
            if install_pytorch_cuda():
                print("\nPlease run this script again to verify GPU access")
    else:
        print("\n" + "="*60)
        print("[OK] YOUR SYSTEM IS READY FOR GPU TRAINING!")
        print("="*60)
        print("\nYou can now run GPU-accelerated training:")
        print("  python gpu_voxel_training.py --model cnn --epochs 100 --batch_size 16")
        print("\nOr use the simple training script:")
        print("  python train_voxel_model.py --model cnn --epochs 100")