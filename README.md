# Subclavian Artery 3D Point Cloud Classification with PointNet

This project implements a deep learning pipeline for classifying 3D subclavian artery models using PointNet neural networks. The pipeline converts STL files to point clouds, generates training labels, and trains a binary classifier.

## Project Overview

- **Dataset**: 104 subclavian artery 3D models in STL format
- **Task**: Binary classification (0 vs 1) of point cloud data
- **Model**: PointNet architecture for 3D point cloud processing
- **Framework**: PyTorch implementation

## Results

- **Test Accuracy**: 56.25% (9/16 correct predictions)
- **Class 0 Accuracy**: 57.14% (4/7 samples)
- **Class 1 Accuracy**: 55.56% (5/9 samples)
- **Training**: 10 epochs on 72 samples
- **Validation**: 16 samples
- **Test**: 16 samples

## Project Structure

```
├── STL/                          # Original STL files (104 files)
├── test_output/                  # PLY point cloud files
├── numpy_arrays/                 # Numpy arrays of point clouds
├── cls/                          # Trained model checkpoints
├── pointnet.pytorch/             # PointNet implementation (submodule)
├── classification_labels.csv     # Random binary labels
├── stl_to_pointcloud.py         # STL to PLY conversion
├── ply_to_numpy.py              # PLY to numpy conversion
├── create_labels.py             # Random label generation
├── subclavian_dataset.py        # Custom PyTorch dataset
├── train_subclavian.py          # Training script
├── test_model.py                # Model testing script
└── README.md                    # This file
```

## Installation

### Prerequisites
- Python 3.7+
- PyTorch
- Required packages:

```bash
pip install torch torchvision pandas numpy trimesh
```

### Setup
1. Clone this repository
2. Clone the PointNet implementation:
```bash
git clone https://github.com/fxia22/pointnet.pytorch.git
```

## Usage

### 1. Data Preprocessing

Convert STL files to point clouds:
```bash
python3 stl_to_pointcloud.py STL -o test_output
```

Convert point clouds to numpy arrays:
```bash
python3 ply_to_numpy.py test_output -o numpy_arrays
```

Generate random classification labels:
```bash
python3 create_labels.py
```

### 2. Training

Train the PointNet model:
```bash
python3 train_subclavian.py --num_points 1024 --nepoch 20 --batchSize 4 --workers 0
```

Parameters:
- `--num_points`: Number of points to sample from each point cloud (default: 2500)
- `--nepoch`: Number of training epochs (default: 250)
- `--batchSize`: Batch size (default: 32)
- `--workers`: Number of data loading workers (use 0 for single process)

### 3. Testing

Test the trained model:
```bash
python3 test_model.py
```

## Dataset Details

### Data Processing Pipeline
1. **STL Files**: 104 3D models of subclavian arteries
2. **Point Cloud Sampling**: 10,000 points sampled from each mesh surface
3. **Normalization**: Points centered and scaled to unit sphere
4. **Data Augmentation**: Random rotation, jitter, and scaling during training

### Dataset Splits
- **Training**: 70% (72 samples)
- **Validation**: 15% (16 samples)  
- **Test**: 15% (16 samples)

### Data Augmentation
- Random rotation around Y-axis
- Gaussian noise (σ=0.02)
- Random scaling (0.8-1.2x)

## Model Architecture

- **Base Model**: PointNet for 3D point cloud classification
- **Input**: Nx3 point clouds (N=1024 points)
- **Output**: Binary classification (2 classes)
- **Features**: Spatial transformer networks, max pooling aggregation

## Training Configuration

- **Optimizer**: Adam (lr=0.001, betas=(0.9, 0.999))
- **Scheduler**: StepLR (step_size=20, gamma=0.5)
- **Loss**: Negative log-likelihood
- **Device**: CPU (CUDA not available)

## Files Description

### Core Scripts
- `stl_to_pointcloud.py`: Converts STL meshes to PLY point clouds using trimesh
- `ply_to_numpy.py`: Converts PLY files to numpy arrays for training
- `create_labels.py`: Generates randomized binary labels for classification
- `subclavian_dataset.py`: Custom PyTorch dataset class with data augmentation
- `train_subclavian.py`: Main training script adapted for subclavian artery data
- `test_model.py`: Model evaluation and testing script

### Data Files
- `classification_labels.csv`: Binary labels for all 104 samples
- `STL/`: Directory containing original 3D model files
- `numpy_arrays/`: Processed point cloud data in numpy format
- `cls/`: Saved model checkpoints from training

## Potential Improvements

1. **More Training Epochs**: Current model trained for only 10 epochs
2. **Hyperparameter Tuning**: Learning rate, batch size, architecture parameters
3. **Better Labels**: Replace random labels with meaningful clinical classifications
4. **Data Augmentation**: Advanced geometric transformations
5. **Model Architecture**: Try PointNet++, DGCNN, or other advanced models
6. **Cross-Validation**: Implement k-fold cross-validation for robust evaluation

## Dependencies

```
torch>=1.6.0
torchvision
pandas
numpy
trimesh
plyfile
```

## Citation

If you use this code, please cite the original PointNet paper:
```
@article{qi2017pointnet,
  title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
  author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
  journal={arXiv preprint arXiv:1612.00593},
  year={2017}
}
```

## License

This project is for educational and research purposes. Please respect the licenses of the underlying libraries and datasets used.