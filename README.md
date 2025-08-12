# 3D Subclavian Artery Classification

A comprehensive machine learning project for classifying subclavian artery 3D models using both traditional ML and deep learning approaches.

## Project Overview

This project implements multiple approaches to classify 3D vessel models (STL files) with anatomical measurements:

- **Traditional ML**: Random Forest, XGBoost, Gradient Boosting with hand-crafted geometric features
- **Deep Learning**: 3D CNN with voxel representation
- **Hybrid Approach**: Multi-modal fusion of point clouds, voxels, and anatomical measurements

## Results Summary

| Model | Cross-Validation Accuracy | Training Time |
|-------|---------------------------|---------------|
| **Random Forest** | 82.98% ± 3.91% | < 1 second |
| **Gradient Boosting** | 82.98% ± 6.12% | < 1 second |
| Hybrid Deep Learning | 79.77% ± 4.03% | ~5 minutes |
| XGBoost | 77.60% ± 4.28% | < 1 second |

## Features

### Traditional ML Approach
- Extracts 51 geometric features from STL files
- Includes volume, surface area, curvature, shape descriptors
- Integrates anatomical measurements (vessel diameters, angles)
- Best performance with small datasets

### Deep Learning Approaches
1. **Voxel-based 3D CNN**: Converts STL to 64x64x64 voxel grids
2. **Hybrid Multi-modal**: Combines:
   - PointNet for point cloud features
   - 3D CNN for voxel features
   - MLP for anatomical measurements
   - Cross-modal attention fusion

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/subclavian-artery-classification.git
cd subclavian-artery-classification

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 1.9+ (with CUDA support recommended)
- scikit-learn
- trimesh
- numpy
- pandas
- matplotlib
- xgboost

## Usage

### 1. Traditional ML Classification
```python
python traditional_ml_approach.py
```

### 2. Hybrid Deep Learning
```python
python hybrid_multimodal_model.py
```

### 3. Cross-Validation Analysis
```python
python cross_validation_analysis.py
```

### 4. Voxel-based CNN
```python
python gpu_voxel_training.py
```

## Data Format

### STL Files
Place 3D vessel models in `STL/` directory

### Labels CSV
Create `classification_labels_with_measurements.csv` with columns:
- `filename`: STL filename (without extension)
- `label`: Binary classification (0 or 1)
- `left_subclavian_diameter_mm`: Vessel diameter
- `aortic_arch_diameter_mm`: Aortic arch diameter
- `angle_degrees`: Anatomical angle

## Project Structure

```
├── STL/                              # 3D model files
├── classification_labels*.csv        # Label files
├── traditional_ml_approach.py        # Traditional ML pipeline
├── hybrid_multimodal_model.py        # Multi-modal deep learning
├── cross_validation_analysis.py      # Model comparison
├── voxel_cnn_model.py               # Voxel-based CNN
├── gpu_voxel_training.py            # GPU-optimized training
├── stl_to_voxel.py                  # STL to voxel conversion
└── requirements.txt                 # Dependencies
```

## Key Findings

1. **Traditional ML performs best** with small datasets (< 100 samples)
2. **Random Forest** offers best stability and speed
3. **Deep learning** requires 500+ samples for optimal performance
4. **Anatomical measurements** are crucial features (23% importance)

## Performance Analysis

With 95 samples:
- Traditional ML achieves ~83% accuracy
- Deep learning limited to ~80% due to insufficient data
- Cross-validation shows 3-6% variance across folds

## Future Improvements

1. **Data Collection**: Target 500+ samples for 90%+ accuracy
2. **Transfer Learning**: Use pre-trained 3D medical models
3. **Ensemble Methods**: Combine multiple approaches
4. **Data Augmentation**: Synthetic data generation

## Citation

If you use this code, please cite:
```
@software{subclavian_classification,
  title = {3D Subclavian Artery Classification},
  year = {2024},
  url = {https://github.com/yourusername/subclavian-artery-classification}
}
```

## License

MIT License - See LICENSE file for details

## Contact

For questions or collaborations, please open an issue on GitHub.
