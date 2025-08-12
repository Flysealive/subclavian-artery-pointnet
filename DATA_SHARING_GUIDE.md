# Data Sharing Guide for Subclavian Artery Classification

This guide explains how to share and set up the large data files that are not included in the GitHub repository.

## Quick Start

### For Data Providers (You)
1. Upload your data folders to Google Drive/Dropbox/OneDrive
2. Update links in `data_config.json`
3. Share the link with collaborators

### For Data Users (Collaborators)
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/subclavian-artery-classification.git
cd subclavian-artery-classification

# Install dependencies
pip install -r requirements.txt

# Run interactive setup
python setup_data.py

# Start using the project
python traditional_ml_approach.py
```

## Data Structure Overview

```
project/
├── STL/                     # 94 STL files (~600 MB)
├── numpy_arrays/            # Preprocessed arrays (~300 MB)
├── voxel_data/             # Voxel representations (~150 MB)
├── hybrid_data/            # Hybrid model data (~400 MB)
│   ├── pointclouds/
│   └── voxels/
├── models/                 # Trained models (~100 MB)
│   ├── best_traditional_ml_model.pkl
│   ├── feature_scaler.pkl
│   └── *.pth files
└── classification_labels_with_measurements.csv
```

## Option 1: Google Drive (Recommended)

### Step 1: Prepare Your Data
```bash
# Create zip files for easier sharing
cd "G:\我的雲端硬碟\1_Projects\AI coding\3D vessel VOXEL\subclavian-artery-pointnet"

# Zip STL files
powershell Compress-Archive -Path STL\* -DestinationPath stl_data.zip

# Zip numpy arrays
powershell Compress-Archive -Path numpy_arrays\* -DestinationPath numpy_arrays.zip

# Zip models
powershell Compress-Archive -Path *.pth,*.pkl -DestinationPath trained_models.zip

# Zip hybrid data
powershell Compress-Archive -Path hybrid_data\* -DestinationPath hybrid_data.zip
```

### Step 2: Upload to Google Drive
1. Upload the zip files to Google Drive
2. Right-click each file → "Get link" → "Anyone with link can view"
3. Copy the sharing links

### Step 3: Update Configuration
Edit `data_config.json` with your Google Drive links:
```json
{
  "data_sources": {
    "google_drive": {
      "individual_files": {
        "stl_data.zip": "https://drive.google.com/file/d/YOUR_FILE_ID/view",
        "trained_models.zip": "https://drive.google.com/file/d/YOUR_FILE_ID/view",
        "numpy_arrays.zip": "https://drive.google.com/file/d/YOUR_FILE_ID/view",
        "hybrid_data.zip": "https://drive.google.com/file/d/YOUR_FILE_ID/view"
      }
    }
  }
}
```

### Step 4: Automated Download
Users can now run:
```bash
python setup_data.py --source gdrive
```

## Option 2: Direct Transfer (Same Network)

### For Provider:
```bash
# Share your project folder over network
# Windows: Right-click folder → Properties → Sharing → Share
```

### For User:
```bash
python setup_data.py --source local --path "\\COMPUTER_NAME\shared_folder\project"
```

## Option 3: Cloud Storage Services

### Dropbox
1. Upload folders to Dropbox
2. Create shared links
3. Update `data_config.json` with Dropbox links

### OneDrive
1. Upload to OneDrive
2. Share → Copy link
3. Update configuration

### AWS S3 / Azure Blob
```python
# Add to setup_data.py for cloud storage
import boto3  # for AWS

def download_from_s3(bucket_name, file_key, local_path):
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, file_key, local_path)
```

## Option 4: USB/External Drive

### Create Portable Package:
```bash
# Run this script to create a portable package
python create_portable_package.py

# This creates subclavian_project_complete.zip with everything
```

### Transfer Process:
1. Copy `subclavian_project_complete.zip` to USB drive
2. On new computer: Extract and run `python setup_data.py`

## Option 5: Git LFS (For GitHub Pro/Enterprise)

```bash
# Initialize Git LFS
git lfs install

# Track large files
git lfs track "*.stl"
git lfs track "*.pth"
git lfs track "*.pkl"

# Add and commit
git add .gitattributes
git add STL/*.stl
git commit -m "Add STL files with LFS"
git push
```

## Regenerating Data from STL Files

If you only have STL files, you can regenerate all preprocessed data:

```bash
# This will generate voxel_data/ and hybrid_data/ from STL files
python setup_data.py --generate
```

## Verification Checklist

After setup, verify your installation:

```bash
# Run verification
python setup_data.py

# Should show:
# ✓ STL files: 94 files ready
# ✓ Labels file found
# ✓ Voxel data: 94 files
# ✓ Found trained models
```

## Troubleshooting

### "No STL files found"
- Check STL/ directory exists
- Verify files have .stl extension
- Run: `python setup_data.py` and choose download option

### "Labels file missing"
- Ensure `classification_labels_with_measurements.csv` is in root directory
- Check it wasn't excluded by .gitignore

### "Import error: trimesh"
```bash
pip install trimesh
```

### "Google Drive download fails"
- Install gdown: `pip install gdown`
- Check file sharing settings (must be "Anyone with link")
- Try direct download and manual extraction

### Large File Issues
- Split into smaller chunks: `split -b 100M large_file.zip`
- Use file compression: 7-Zip for better compression
- Consider cloud transfer services: WeTransfer, SendGB

## Data Privacy Note

If your STL files contain sensitive medical data:
1. Use private Google Drive links
2. Add password protection to zip files
3. Consider encryption: `7z a -p"password" secure_data.7z STL/`
4. Use secure transfer methods
5. Add data usage agreement to repository

## Contact for Data Access

If you need access to the original dataset:
- Open an issue on GitHub
- Email: [your-email@example.com]
- Include your intended use case and affiliation

## File Size Reference

| Component | File Count | Total Size | Essential? |
|-----------|------------|------------|------------|
| STL files | 94 | ~600 MB | Yes |
| Labels CSV | 2 | <1 MB | Yes |
| Numpy arrays | 94 | ~300 MB | No (can regenerate) |
| Voxel data | 94 | ~150 MB | No (can regenerate) |
| Hybrid data | 188 | ~400 MB | No (can regenerate) |
| Trained models | 5-10 | ~100 MB | No (can retrain) |

**Minimum required**: STL files + Labels CSV (~601 MB)
**Full package**: All components (~1.5 GB)