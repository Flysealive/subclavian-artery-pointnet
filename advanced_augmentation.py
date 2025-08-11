#!/usr/bin/env python3
"""
Advanced Point Cloud Augmentation for Medical Data
Implements Claude Sonnet's suggestions adapted for subclavian artery classification
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional, Dict, Any
import torch

class MedicalPointCloudAugmentor:
    """
    Advanced augmentation for medical point clouds
    Balances aggressive augmentation with anatomical preservation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize augmentor with configuration
        
        Args:
            config: Dictionary with augmentation parameters
        """
        config = config or {}
        
        # Geometric transformations
        self.rotation_range = config.get('rotation_range', 30)  # degrees
        self.translation_range = config.get('translation_range', 0.1)
        self.scale_range = config.get('scale_range', (0.9, 1.1))
        self.shear_range = config.get('shear_range', 0.1)
        
        # Point-level modifications
        self.jitter_sigma = config.get('jitter_sigma', 0.01)
        self.jitter_clip = config.get('jitter_clip', 0.05)
        self.dropout_ratio = config.get('dropout_ratio', 0.1)
        self.outlier_ratio = config.get('outlier_ratio', 0.02)
        
        # Advanced techniques
        self.elastic_alpha = config.get('elastic_alpha', 1.0)
        self.elastic_sigma = config.get('elastic_sigma', 0.1)
        self.cutout_ratio = config.get('cutout_ratio', 0.2)
        
        # Class-specific settings
        self.minority_boost = config.get('minority_boost', 2.0)  # Stronger aug for minority class
        
    # ============= GEOMETRIC TRANSFORMATIONS =============
    
    def random_rotation(self, points: np.ndarray, axis: str = 'y') -> np.ndarray:
        """Random rotation around specified axis"""
        angle = np.random.uniform(-self.rotation_range, self.rotation_range) * np.pi / 180
        
        if axis == 'x':
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
        elif axis == 'y':
            rotation_matrix = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])
        else:  # z-axis
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
        
        return points @ rotation_matrix
    
    def random_rotation_3d(self, points: np.ndarray) -> np.ndarray:
        """Full 3D rotation with random Euler angles"""
        angles = np.random.uniform(-self.rotation_range, self.rotation_range, 3) * np.pi / 180
        
        # Rotation matrices for each axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])]
        ])
        
        Ry = np.array([
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])]
        ])
        
        Rz = np.array([
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation
        R = Rz @ Ry @ Rx
        return points @ R
    
    def random_translation(self, points: np.ndarray) -> np.ndarray:
        """Random translation within bounds"""
        shift = np.random.uniform(-self.translation_range, self.translation_range, 3)
        return points + shift
    
    def random_scale(self, points: np.ndarray, uniform: bool = True) -> np.ndarray:
        """Random scaling (uniform or non-uniform)"""
        if uniform:
            scale = np.random.uniform(*self.scale_range)
            return points * scale
        else:
            # Non-uniform scaling for each axis
            scales = np.random.uniform(*self.scale_range, size=3)
            return points * scales
    
    def random_shear(self, points: np.ndarray) -> np.ndarray:
        """Apply random shear transformation"""
        shear_matrix = np.eye(3)
        # Add shear components
        shear_matrix[0, 1] = np.random.uniform(-self.shear_range, self.shear_range)
        shear_matrix[0, 2] = np.random.uniform(-self.shear_range, self.shear_range)
        shear_matrix[1, 2] = np.random.uniform(-self.shear_range, self.shear_range)
        return points @ shear_matrix
    
    def random_reflection(self, points: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """Random reflection along axis"""
        if axis is None:
            axis = np.random.randint(0, 3)
        
        reflection_matrix = np.eye(3)
        reflection_matrix[axis, axis] = -1
        return points @ reflection_matrix
    
    # ============= POINT-LEVEL MODIFICATIONS =============
    
    def jittering(self, points: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to points"""
        noise = np.clip(
            np.random.normal(0, self.jitter_sigma, points.shape),
            -self.jitter_clip, self.jitter_clip
        )
        return points + noise
    
    def random_dropout(self, points: np.ndarray, maintain_size: bool = True) -> np.ndarray:
        """Randomly remove points"""
        n_points = points.shape[0]
        n_keep = int(n_points * (1 - self.dropout_ratio))
        
        if n_keep < n_points:
            choice = np.random.choice(n_points, n_keep, replace=False)
            kept_points = points[choice, :]
            
            if maintain_size:
                # Resample to maintain original size
                resample_idx = np.random.choice(kept_points.shape[0], n_points, replace=True)
                return kept_points[resample_idx, :]
            else:
                return kept_points
        return points
    
    def random_sampling(self, points: np.ndarray, n_samples: Optional[int] = None) -> np.ndarray:
        """Random subsampling with varying density"""
        if n_samples is None:
            # Random sampling ratio
            ratio = np.random.uniform(0.5, 1.0)
            n_samples = int(points.shape[0] * ratio)
        
        if n_samples < points.shape[0]:
            choice = np.random.choice(points.shape[0], n_samples, replace=False)
        else:
            choice = np.random.choice(points.shape[0], n_samples, replace=True)
        
        return points[choice, :]
    
    def outlier_injection(self, points: np.ndarray) -> np.ndarray:
        """Add random outlier points"""
        n_outliers = int(points.shape[0] * self.outlier_ratio)
        if n_outliers > 0:
            # Generate outliers within extended bounds
            bounds_min = points.min(axis=0) * 1.5
            bounds_max = points.max(axis=0) * 1.5
            outliers = np.random.uniform(bounds_min, bounds_max, (n_outliers, 3))
            points = np.vstack([points, outliers])
        return points
    
    # ============= ADVANCED TECHNIQUES =============
    
    def elastic_deformation(self, points: np.ndarray) -> np.ndarray:
        """Apply smooth elastic deformation"""
        # Create random displacement field
        displacement = np.random.randn(*points.shape) * self.elastic_sigma
        
        # Smooth the displacement field
        for i in range(3):
            displacement[:, i] = gaussian_filter(displacement[:, i], sigma=self.elastic_alpha)
        
        return points + displacement
    
    def cutout_augmentation(self, points: np.ndarray, maintain_size: bool = True) -> np.ndarray:
        """Remove a region of points (simulates occlusion)"""
        n_points = points.shape[0]
        
        # Select random center point
        center_idx = np.random.randint(0, points.shape[0])
        center = points[center_idx]
        
        # Calculate distances from center
        distances = np.linalg.norm(points - center, axis=1)
        
        # Determine cutout radius
        radius = np.percentile(distances, self.cutout_ratio * 100)
        
        # Keep points outside radius
        mask = distances > radius
        kept_points = points[mask]
        
        if maintain_size and kept_points.shape[0] < n_points:
            # Resample to maintain original size
            if kept_points.shape[0] > 0:
                resample_idx = np.random.choice(kept_points.shape[0], n_points, replace=True)
                return kept_points[resample_idx, :]
            else:
                return points  # Return original if all points removed
        
        return kept_points if kept_points.shape[0] > 0 else points
    
    def gravity_alignment(self, points: np.ndarray, gravity_axis: int = 2) -> np.ndarray:
        """Align object with simulated gravity direction"""
        # First apply random rotation
        points = self.random_rotation_3d(points)
        
        # Find principal axis using PCA
        centered = points - points.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Align smallest eigenvalue direction with gravity
        min_idx = np.argmin(eigenvalues)
        gravity_vector = np.zeros(3)
        gravity_vector[gravity_axis] = 1
        
        # Compute rotation to align
        v = eigenvectors[:, min_idx]
        axis = np.cross(v, gravity_vector)
        angle = np.arccos(np.clip(np.dot(v, gravity_vector), -1, 1))
        
        if np.linalg.norm(axis) > 1e-6:
            axis = axis / np.linalg.norm(axis)
            # Rodrigues' rotation formula
            K = np.array([[0, -axis[2], axis[1]],
                         [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
            points = points @ R.T
        
        return points
    
    # ============= MEDICAL-SPECIFIC AUGMENTATIONS =============
    
    def anatomical_preserving_rotation(self, points: np.ndarray) -> np.ndarray:
        """Limited rotation that preserves anatomical orientation"""
        # For medical data, use smaller rotation ranges
        angle = np.random.uniform(-15, 15) * np.pi / 180  # Max 15 degrees
        
        # Primarily rotate around superior-inferior axis (typically z)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        return points @ rotation_matrix
    
    def scan_noise_simulation(self, points: np.ndarray) -> np.ndarray:
        """Simulate medical scanner measurement noise"""
        # Distance-dependent noise (farther from center = more noise)
        center = points.mean(axis=0)
        distances = np.linalg.norm(points - center, axis=1, keepdims=True)
        
        # Normalize distances
        max_dist = distances.max()
        if max_dist > 0:
            noise_scale = (distances / max_dist) * 0.01  # Max 1% noise at edges
            noise = np.random.randn(*points.shape) * noise_scale
            points = points + noise
        
        return points
    
    # ============= AUGMENTATION PIPELINE =============
    
    def augment(self, 
                points: np.ndarray, 
                label: int,
                aggressive: bool = False) -> Tuple[np.ndarray, int]:
        """
        Apply augmentation pipeline
        
        Args:
            points: Input point cloud (N, 3)
            label: Class label
            aggressive: Whether to apply aggressive augmentation
        
        Returns:
            Augmented points and label
        """
        # Store original for mixing if needed
        original_points = points.copy()
        
        # Determine augmentation strength based on class
        is_minority = (label == 1)  # Assuming 1 is minority class
        prob_multiplier = self.minority_boost if is_minority else 1.0
        
        # Geometric transformations
        if np.random.rand() < 0.8 * prob_multiplier:
            if aggressive:
                points = self.random_rotation_3d(points)
            else:
                points = self.anatomical_preserving_rotation(points)
        
        if np.random.rand() < 0.5 * prob_multiplier:
            points = self.random_translation(points)
        
        if np.random.rand() < 0.5 * prob_multiplier:
            points = self.random_scale(points, uniform=True)
        
        if aggressive and np.random.rand() < 0.3 * prob_multiplier:
            points = self.random_shear(points)
        
        # Point-level modifications
        if np.random.rand() < 0.7 * prob_multiplier:
            points = self.jittering(points)
        
        if np.random.rand() < 0.3 * prob_multiplier:
            points = self.random_dropout(points, maintain_size=True)
        
        # Skip outlier injection as it changes point count
        # if aggressive and np.random.rand() < 0.2 * prob_multiplier:
        #     points = self.outlier_injection(points)
        
        # Advanced techniques (use sparingly for medical data)
        if aggressive and np.random.rand() < 0.2 * prob_multiplier:
            points = self.elastic_deformation(points)
        
        if aggressive and np.random.rand() < 0.15 * prob_multiplier:
            points = self.cutout_augmentation(points, maintain_size=True)
        
        # Medical-specific
        if np.random.rand() < 0.5:
            points = self.scan_noise_simulation(points)
        
        return points, label
    
    def mixup(self, 
              points1: np.ndarray, 
              points2: np.ndarray,
              label1: int, 
              label2: int,
              alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        MixUp augmentation for point clouds
        
        Args:
            points1, points2: Two point clouds
            label1, label2: Corresponding labels
            alpha: Beta distribution parameter
        
        Returns:
            Mixed points and soft label
        """
        lam = np.random.beta(alpha, alpha)
        
        # Ensure same number of points
        min_points = min(points1.shape[0], points2.shape[0])
        points1 = points1[:min_points]
        points2 = points2[:min_points]
        
        # Mix points
        mixed_points = lam * points1 + (1 - lam) * points2
        
        # Mix labels (soft labels for training)
        mixed_label = np.array([0.0, 0.0])
        mixed_label[label1] += lam
        mixed_label[label2] += (1 - lam)
        
        return mixed_points, mixed_label


def test_augmentations():
    """Test and visualize augmentations"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create sample point cloud (sphere)
    n_points = 1000
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    points = np.stack([x, y, z], axis=1)
    
    # Initialize augmentor
    config = {
        'rotation_range': 45,
        'scale_range': (0.8, 1.2),
        'jitter_sigma': 0.02,
        'dropout_ratio': 0.15
    }
    augmentor = MedicalPointCloudAugmentor(config)
    
    # Apply different augmentations
    fig = plt.figure(figsize=(15, 10))
    
    augmentations = [
        ('Original', points),
        ('Rotation', augmentor.random_rotation_3d(points.copy())),
        ('Jittering', augmentor.jittering(points.copy())),
        ('Dropout', augmentor.random_dropout(points.copy())),
        ('Elastic', augmentor.elastic_deformation(points.copy())),
        ('Cutout', augmentor.cutout_augmentation(points.copy())),
        ('Full Pipeline', augmentor.augment(points.copy(), label=0, aggressive=True)[0])
    ]
    
    for i, (title, aug_points) in enumerate(augmentations):
        ax = fig.add_subplot(2, 4, i+1, projection='3d')
        ax.scatter(aug_points[:, 0], aug_points[:, 1], aug_points[:, 2], 
                  c=aug_points[:, 2], cmap='viridis', s=1)
        ax.set_title(title)
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
    
    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=150)
    print("Augmentation examples saved to augmentation_examples.png")


if __name__ == "__main__":
    print("Advanced Point Cloud Augmentation Module")
    print("=" * 50)
    print("\nFeatures implemented (from Claude Sonnet's suggestions):")
    print("✓ Geometric Transformations (rotation, translation, scaling, shearing, reflection)")
    print("✓ Point-Level Modifications (jittering, dropout, sampling, outliers)")
    print("✓ Advanced Techniques (elastic deformation, cutout, gravity alignment)")
    print("✓ Medical-Specific (anatomical preservation, scan noise)")
    print("✓ Class-Specific Boosting (stronger augmentation for minority class)")
    print("✓ MixUp augmentation")
    
    print("\nTesting augmentations...")
    test_augmentations()
    
    print("\nUsage example:")
    print("-" * 50)
    print("""
    from advanced_augmentation import MedicalPointCloudAugmentor
    
    # Initialize with custom config
    config = {
        'rotation_range': 30,  # degrees
        'minority_boost': 2.0,  # 2x stronger aug for minority class
        'jitter_sigma': 0.01
    }
    augmentor = MedicalPointCloudAugmentor(config)
    
    # In your dataset __getitem__:
    if self.training:
        points, label = augmentor.augment(points, label, aggressive=False)
    """)
    
    print("\nIntegration complete!")