"""
Data Setup Script for Subclavian Artery Classification Project
===============================================================
This script helps set up all necessary data files when cloning the project
on a new computer or environment.

Usage:
    python setup_data.py                    # Interactive setup
    python setup_data.py --source gdrive    # Download from Google Drive
    python setup_data.py --source local     # Copy from local path
    python setup_data.py --generate         # Generate from STL files
"""

import os
import sys
import json
import shutil
import zipfile
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Optional imports (will check if available)
try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False
    
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

class DataSetup:
    """Handles complete data setup for the project"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        
        # Define expected directory structure
        self.directories = {
            'STL': 'STL files (3D models)',
            'numpy_arrays': 'Preprocessed numpy arrays',
            'voxel_data': 'Voxel representations',
            'hybrid_data': 'Hybrid data (point clouds + voxels)',
            'hybrid_data/pointclouds': 'Point cloud data',
            'hybrid_data/voxels': 'Voxel data for hybrid model',
            'models': 'Saved model files',
            'logs': 'Training logs',
            'test_output': 'Test outputs',
            'standardized_data': 'Standardized datasets'
        }
        
        # Define expected files
        self.required_files = {
            'classification_labels.csv': 'Basic labels',
            'classification_labels_with_measurements.csv': 'Labels with anatomical measurements',
            'requirements.txt': 'Python dependencies'
        }
        
        # Data source configuration (UPDATE THESE WITH YOUR LINKS)
        self.gdrive_links = {
            'stl_data.zip': 'https://drive.google.com/file/d/YOUR_FILE_ID_1/view?usp=sharing',
            'numpy_arrays.zip': 'https://drive.google.com/file/d/YOUR_FILE_ID_2/view?usp=sharing',
            'trained_models.zip': 'https://drive.google.com/file/d/YOUR_FILE_ID_3/view?usp=sharing',
            'hybrid_data.zip': 'https://drive.google.com/file/d/YOUR_FILE_ID_4/view?usp=sharing'
        }
        
        # Alternative: Direct download links (if using other storage)
        self.direct_links = {
            'stl_data.zip': 'https://your-storage.com/stl_data.zip',
            'numpy_arrays.zip': 'https://your-storage.com/numpy_arrays.zip',
            'trained_models.zip': 'https://your-storage.com/models.zip'
        }
        
    def check_environment(self):
        """Check Python environment and dependencies"""
        print("\n=== Environment Check ===")
        print(f"Python version: {sys.version}")
        print(f"Current directory: {os.getcwd()}")
        
        # Check for key dependencies
        dependencies = {
            'numpy': 'NumPy',
            'torch': 'PyTorch',
            'sklearn': 'Scikit-learn',
            'trimesh': 'Trimesh (for STL processing)',
            'pandas': 'Pandas',
            'gdown': 'Gdown (for Google Drive downloads)'
        }
        
        missing = []
        for module, name in dependencies.items():
            try:
                __import__(module)
                print(f"✓ {name} installed")
            except ImportError:
                print(f"✗ {name} not installed")
                missing.append(module)
        
        if missing:
            print(f"\nMissing packages: {', '.join(missing)}")
            print("Install with: pip install -r requirements.txt")
            
        return len(missing) == 0
    
    def setup_directories(self):
        """Create all necessary directories"""
        print("\n=== Setting Up Directories ===")
        
        for dir_path, description in self.directories.items():
            path = self.project_root / dir_path
            path.mkdir(parents=True, exist_ok=True)
            
            # Check if directory has content
            content_count = len(list(path.glob('*')))
            if content_count > 0:
                print(f"✓ {dir_path}/ exists ({content_count} items)")
            else:
                print(f"✓ {dir_path}/ created (empty)")
    
    def check_existing_data(self):
        """Check what data already exists"""
        print("\n=== Checking Existing Data ===")
        
        status = {
            'stl_files': 0,
            'numpy_arrays': 0,
            'voxel_files': 0,
            'models': 0,
            'labels': False
        }
        
        # Check STL files
        stl_path = self.project_root / 'STL'
        stl_files = list(stl_path.glob('*.stl'))
        status['stl_files'] = len(stl_files)
        print(f"STL files: {status['stl_files']} found")
        
        # Check numpy arrays
        numpy_path = self.project_root / 'numpy_arrays'
        numpy_files = list(numpy_path.glob('*.npy'))
        status['numpy_arrays'] = len(numpy_files)
        print(f"Numpy arrays: {status['numpy_arrays']} found")
        
        # Check voxel data
        voxel_path = self.project_root / 'voxel_data'
        voxel_files = list(voxel_path.glob('*.npy'))
        status['voxel_files'] = len(voxel_files)
        print(f"Voxel files: {status['voxel_files']} found")
        
        # Check models
        model_extensions = ['*.pth', '*.pkl', '*.h5']
        model_files = []
        for ext in model_extensions:
            model_files.extend(list(self.project_root.glob(ext)))
        status['models'] = len(model_files)
        print(f"Trained models: {status['models']} found")
        
        # Check label files
        labels_path = self.project_root / 'classification_labels_with_measurements.csv'
        if labels_path.exists():
            status['labels'] = True
            print(f"✓ Labels file found")
        else:
            print(f"✗ Labels file missing")
        
        return status
    
    def download_from_gdrive(self):
        """Download data from Google Drive"""
        if not GDOWN_AVAILABLE:
            print("\n✗ gdown not installed. Install with: pip install gdown")
            return False
        
        print("\n=== Downloading from Google Drive ===")
        print("Note: Update gdrive_links in this script with your actual Google Drive links")
        
        for filename, url in self.gdrive_links.items():
            if 'YOUR_FILE_ID' in url:
                print(f"⚠ Skipping {filename}: Update Google Drive link in setup_data.py")
                continue
                
            output_path = self.project_root / filename
            if output_path.exists():
                print(f"✓ {filename} already exists")
                continue
            
            print(f"Downloading {filename}...")
            try:
                # Extract file ID from Google Drive URL
                if '/d/' in url:
                    file_id = url.split('/d/')[1].split('/')[0]
                    gdown.download(f'https://drive.google.com/uc?id={file_id}', 
                                 str(output_path), quiet=False)
                    
                    # Extract if it's a zip file
                    if filename.endswith('.zip'):
                        self.extract_zip(output_path)
                    print(f"✓ Downloaded {filename}")
                else:
                    print(f"✗ Invalid Google Drive URL for {filename}")
            except Exception as e:
                print(f"✗ Failed to download {filename}: {e}")
                
        return True
    
    def extract_zip(self, zip_path: Path):
        """Extract a zip file and remove it"""
        print(f"Extracting {zip_path.name}...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract to appropriate directory based on zip name
                if 'stl' in zip_path.name.lower():
                    extract_to = self.project_root / 'STL'
                elif 'numpy' in zip_path.name.lower():
                    extract_to = self.project_root / 'numpy_arrays'
                elif 'model' in zip_path.name.lower():
                    extract_to = self.project_root / 'models'
                elif 'hybrid' in zip_path.name.lower():
                    extract_to = self.project_root / 'hybrid_data'
                else:
                    extract_to = self.project_root
                
                zip_ref.extractall(extract_to)
                print(f"✓ Extracted to {extract_to}")
            
            # Remove zip file after extraction
            zip_path.unlink()
            print(f"✓ Removed {zip_path.name}")
            
        except Exception as e:
            print(f"✗ Failed to extract {zip_path.name}: {e}")
    
    def copy_from_local(self, source_path: str):
        """Copy data from a local path"""
        source = Path(source_path)
        
        if not source.exists():
            print(f"✗ Source path does not exist: {source}")
            return False
        
        print(f"\n=== Copying from {source} ===")
        
        # Copy STL files
        stl_source = source / 'STL'
        if stl_source.exists():
            stl_files = list(stl_source.glob('*.stl'))
            if stl_files:
                print(f"Copying {len(stl_files)} STL files...")
                for stl_file in stl_files:
                    dest = self.project_root / 'STL' / stl_file.name
                    if not dest.exists():
                        shutil.copy2(stl_file, dest)
                print(f"✓ Copied STL files")
        
        # Copy other directories
        for dir_name in ['numpy_arrays', 'voxel_data', 'hybrid_data', 'models']:
            source_dir = source / dir_name
            if source_dir.exists():
                dest_dir = self.project_root / dir_name
                print(f"Copying {dir_name}...")
                shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
                print(f"✓ Copied {dir_name}")
        
        # Copy model files in root
        for pattern in ['*.pth', '*.pkl', '*.h5']:
            files = list(source.glob(pattern))
            for file in files:
                dest = self.project_root / file.name
                if not dest.exists():
                    shutil.copy2(file, dest)
                    print(f"✓ Copied {file.name}")
        
        return True
    
    def generate_preprocessed_data(self):
        """Generate voxel and numpy data from STL files"""
        if not TRIMESH_AVAILABLE:
            print("\n✗ trimesh not installed. Install with: pip install trimesh")
            return False
        
        print("\n=== Generating Preprocessed Data ===")
        
        stl_files = list((self.project_root / 'STL').glob('*.stl'))
        if not stl_files:
            print("✗ No STL files found in STL/ directory")
            return False
        
        print(f"Found {len(stl_files)} STL files")
        
        # Import processing functions
        try:
            from stl_to_voxel import stl_to_voxel_pipeline
            print("✓ Imported voxel conversion functions")
            
            # Process STL files
            print("Converting STL files to voxels...")
            for i, stl_file in enumerate(stl_files, 1):
                output_path = self.project_root / 'voxel_data' / f"{stl_file.stem}.npy"
                if not output_path.exists():
                    print(f"  [{i}/{len(stl_files)}] Processing {stl_file.name}...")
                    voxel_data = stl_to_voxel_pipeline(str(stl_file))
                    np.save(output_path, voxel_data)
            
            print("✓ Generated voxel data")
            
        except ImportError as e:
            print(f"✗ Could not import processing functions: {e}")
            print("  Make sure stl_to_voxel.py is in the project directory")
            return False
        
        return True
    
    def create_data_manifest(self):
        """Create a manifest file listing all data files"""
        print("\n=== Creating Data Manifest ===")
        
        manifest = {
            'created': str(Path.cwd()),
            'directories': {},
            'files': {}
        }
        
        # List contents of each directory
        for dir_name in self.directories.keys():
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                files = list(dir_path.glob('*'))
                manifest['directories'][dir_name] = {
                    'exists': True,
                    'file_count': len(files),
                    'total_size_mb': sum(f.stat().st_size for f in files if f.is_file()) / (1024*1024)
                }
        
        # Check for important files
        important_files = [
            'classification_labels.csv',
            'classification_labels_with_measurements.csv',
            'best_model.pth',
            'best_traditional_ml_model.pkl',
            'feature_scaler.pkl',
            'best_hybrid_model.pth',
            'best_hybrid_150epochs.pth'
        ]
        
        for filename in important_files:
            file_path = self.project_root / filename
            if file_path.exists():
                manifest['files'][filename] = {
                    'exists': True,
                    'size_mb': file_path.stat().st_size / (1024*1024)
                }
        
        # Save manifest
        manifest_path = self.project_root / 'data_manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"✓ Created data_manifest.json")
        return manifest
    
    def verify_setup(self):
        """Verify the setup is complete and ready"""
        print("\n=== Verifying Setup ===")
        
        ready = True
        
        # Check STL files
        stl_files = list((self.project_root / 'STL').glob('*.stl'))
        if len(stl_files) >= 90:
            print(f"✓ STL files: {len(stl_files)} files ready")
        else:
            print(f"⚠ STL files: Only {len(stl_files)} files (expected ~94)")
            ready = False
        
        # Check labels
        labels_file = self.project_root / 'classification_labels_with_measurements.csv'
        if labels_file.exists():
            print("✓ Labels file found")
        else:
            print("✗ Labels file missing")
            ready = False
        
        # Check for at least one model
        model_files = list(self.project_root.glob('*.pth')) + list(self.project_root.glob('*.pkl'))
        if model_files:
            print(f"✓ Found {len(model_files)} trained models")
        else:
            print("⚠ No trained models found (optional)")
        
        # Check voxel data
        voxel_files = list((self.project_root / 'voxel_data').glob('*.npy'))
        if voxel_files:
            print(f"✓ Voxel data: {len(voxel_files)} files")
        else:
            print("⚠ No voxel data (will be generated if needed)")
        
        if ready:
            print("\n✅ Setup complete! You can now run the training scripts:")
            print("  - python traditional_ml_approach.py")
            print("  - python hybrid_multimodal_model.py")
            print("  - python cross_validation_analysis.py")
        else:
            print("\n⚠ Setup incomplete. Some files are missing.")
            print("  You can still run the scripts, but may need to generate data.")
        
        return ready
    
    def interactive_setup(self):
        """Interactive setup wizard"""
        print("\n" + "="*60)
        print("SUBCLAVIAN ARTERY CLASSIFICATION - DATA SETUP")
        print("="*60)
        
        # Check environment
        env_ok = self.check_environment()
        if not env_ok:
            response = input("\nSome dependencies are missing. Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
        
        # Setup directories
        self.setup_directories()
        
        # Check existing data
        status = self.check_existing_data()
        
        # Determine what's needed
        need_stl = status['stl_files'] < 90
        need_labels = not status['labels']
        
        if need_stl or need_labels:
            print("\n=== Data Download Options ===")
            print("1. Download from Google Drive (requires gdown)")
            print("2. Copy from local path")
            print("3. Manual setup (I'll download/copy files myself)")
            print("4. Generate preprocessed data from existing STL files")
            
            choice = input("\nSelect option (1-4): ")
            
            if choice == '1':
                self.download_from_gdrive()
            elif choice == '2':
                source = input("Enter source path (e.g., D:\\backup\\project): ")
                self.copy_from_local(source)
            elif choice == '3':
                print("\n=== Manual Setup Instructions ===")
                print("1. Copy your STL files to: STL/")
                print("2. Copy classification_labels_with_measurements.csv to project root")
                print("3. (Optional) Copy trained models (*.pth, *.pkl) to project root")
                print("4. Run this script again to verify")
            elif choice == '4':
                if status['stl_files'] > 0:
                    self.generate_preprocessed_data()
                else:
                    print("✗ No STL files found to process")
        
        # Create manifest
        self.create_data_manifest()
        
        # Final verification
        self.verify_setup()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Setup data for Subclavian Artery Classification')
    parser.add_argument('--source', choices=['gdrive', 'local', 'generate'], 
                       help='Data source method')
    parser.add_argument('--path', type=str, 
                       help='Local path for copying data')
    parser.add_argument('--generate', action='store_true',
                       help='Generate preprocessed data from STL files')
    
    args = parser.parse_args()
    
    setup = DataSetup()
    
    if args.generate:
        setup.setup_directories()
        setup.generate_preprocessed_data()
        setup.verify_setup()
    elif args.source == 'gdrive':
        setup.setup_directories()
        setup.download_from_gdrive()
        setup.verify_setup()
    elif args.source == 'local' and args.path:
        setup.setup_directories()
        setup.copy_from_local(args.path)
        setup.verify_setup()
    else:
        # Interactive mode
        setup.interactive_setup()

if __name__ == "__main__":
    main()