#!/usr/bin/env python3

import numpy as np
import trimesh
from pathlib import Path
import pandas as pd

class ImprovedSTLProcessor:
    """
    Enhanced STL processing with better normalization and quality control
    """
    
    def __init__(self, target_points=2048, quality_checks=True):
        self.target_points = target_points
        self.quality_checks = quality_checks
    
    def process_stl_file(self, stl_path):
        """
        Process a single STL file with enhanced normalization
        """
        try:
            # Load mesh
            mesh = trimesh.load(str(stl_path))
            
            # Quality checks
            if self.quality_checks:
                if not self.check_mesh_quality(mesh):
                    print(f"Warning: Quality issues detected in {stl_path.name}")
            
            # Enhanced sampling strategy
            points = self.enhanced_sampling(mesh)
            
            # Advanced normalization
            points = self.advanced_normalization(points)
            
            return points, True
            
        except Exception as e:
            print(f"Error processing {stl_path.name}: {e}")
            return None, False
    
    def check_mesh_quality(self, mesh):
        """
        Check for common mesh quality issues
        """
        issues = []
        
        # Check for degenerate faces
        if hasattr(mesh, 'faces'):
            face_areas = mesh.area_faces
            degenerate_faces = np.sum(face_areas < 1e-6)
            if degenerate_faces > len(mesh.faces) * 0.01:  # More than 1% degenerate
                issues.append(f"High degenerate faces: {degenerate_faces}")
        
        # Check for non-manifold edges
        if hasattr(mesh, 'is_watertight') and not mesh.is_watertight:
            issues.append("Non-watertight mesh")
        
        # Check for extreme aspect ratios
        bounds = mesh.bounds
        dimensions = bounds[1] - bounds[0]
        if np.max(dimensions) / np.min(dimensions) > 10:
            issues.append("Extreme aspect ratio")
        
        if issues:
            print(f"  Mesh issues: {', '.join(issues)}")
            return False
        return True
    
    def enhanced_sampling(self, mesh):
        """
        Improved point sampling strategy
        """
        # Strategy 1: Surface sampling with multiple methods
        points_list = []
        
        # Method 1: Uniform surface sampling (70% of points)
        n_uniform = int(self.target_points * 0.7)
        uniform_points = mesh.sample(n_uniform)
        points_list.append(uniform_points)
        
        # Method 2: Curvature-aware sampling (20% of points)
        n_curvature = int(self.target_points * 0.2)
        try:
            # Sample more points from high-curvature areas
            vertex_normals = mesh.vertex_normals
            curvature_weights = np.linalg.norm(np.gradient(vertex_normals, axis=0), axis=1)
            curvature_weights = curvature_weights / curvature_weights.sum()
            
            curvature_indices = np.random.choice(
                len(mesh.vertices), 
                size=n_curvature, 
                p=curvature_weights,
                replace=True
            )
            curvature_points = mesh.vertices[curvature_indices]
            points_list.append(curvature_points)
        except:
            # Fallback to uniform sampling
            fallback_points = mesh.sample(n_curvature)
            points_list.append(fallback_points)
        
        # Method 3: Edge-aware sampling (10% of points)
        n_edge = self.target_points - n_uniform - n_curvature
        try:
            # Sample near mesh edges
            edges = mesh.edges_unique
            edge_midpoints = (mesh.vertices[edges[:, 0]] + mesh.vertices[edges[:, 1]]) / 2
            
            if len(edge_midpoints) >= n_edge:
                edge_indices = np.random.choice(len(edge_midpoints), size=n_edge, replace=False)
                edge_points = edge_midpoints[edge_indices]
            else:
                # If not enough edges, repeat existing edge points
                edge_indices = np.random.choice(len(edge_midpoints), size=n_edge, replace=True)
                edge_points = edge_midpoints[edge_indices]
            points_list.append(edge_points)
        except:
            # Fallback to uniform sampling
            fallback_points = mesh.sample(n_edge)
            points_list.append(fallback_points)
        
        # Combine all points
        all_points = np.vstack(points_list)
        
        # Shuffle to mix different sampling methods
        indices = np.random.permutation(len(all_points))
        return all_points[indices]
    
    def advanced_normalization(self, points):
        """
        Enhanced normalization for better consistency
        """
        # Step 1: Remove outliers using statistical methods
        points = self.remove_statistical_outliers(points)
        
        # Step 2: Center using robust statistics (median instead of mean)
        centroid = np.median(points, axis=0)
        points = points - centroid
        
        # Step 3: Scale using robust scaling (IQR instead of std)
        distances = np.linalg.norm(points, axis=1)
        q75 = np.percentile(distances, 75)
        q25 = np.percentile(distances, 25)
        iqr_scale = q75 - q25
        
        if iqr_scale > 1e-6:  # Avoid division by zero
            scale_factor = 1.0 / iqr_scale
        else:
            scale_factor = 1.0 / np.max(distances) if np.max(distances) > 1e-6 else 1.0
        
        points = points * scale_factor
        
        # Step 4: Optional PCA alignment for consistent orientation
        points = self.align_principal_components(points)
        
        return points
    
    def remove_statistical_outliers(self, points, std_threshold=2.0):
        """
        Remove statistical outliers based on distance from centroid
        """
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # Keep points within threshold standard deviations
        mask = distances < (mean_dist + std_threshold * std_dist)
        return points[mask]
    
    def align_principal_components(self, points):
        """
        Align point cloud with principal components for consistent orientation
        """
        try:
            # Compute PCA
            cov_matrix = np.cov(points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Sort by eigenvalue magnitude (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            
            # Ensure consistent handedness
            if np.linalg.det(eigenvectors) < 0:
                eigenvectors[:, -1] *= -1
            
            # Rotate points to align with principal components
            aligned_points = points @ eigenvectors
            
            return aligned_points
        except:
            # If PCA fails, return original points
            return points

def create_improved_dataset(stl_dir, output_dir, dataset_csv=None):
    """
    Create improved point cloud dataset with enhanced preprocessing
    """
    processor = ImprovedSTLProcessor(target_points=2048)
    
    stl_path = Path(stl_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Get list of files to process
    if dataset_csv:
        df = pd.read_csv(dataset_csv)
        files_to_process = [stl_path / filename for filename in df['filename']]
        print(f"Processing {len(files_to_process)} files from {dataset_csv}")
    else:
        files_to_process = list(stl_path.glob("*.stl"))
        print(f"Processing all {len(files_to_process)} STL files")
    
    successful = 0
    failed = 0
    
    for stl_file in files_to_process:
        points, success = processor.process_stl_file(stl_file)
        
        if success and points is not None:
            # Save as numpy array
            output_file = output_path / f"{stl_file.stem}.npy"
            np.save(output_file, points.astype(np.float32))
            successful += 1
        else:
            failed += 1
        
        if (successful + failed) % 10 == 0:
            print(f"Processed: {successful + failed}/{len(files_to_process)}")
    
    print(f"\nProcessing complete!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output directory: {output_path}")

if __name__ == "__main__":
    # Create improved datasets for different quality levels
    
    # Process high-quality subset if available
    if Path('standardized_data/high_quality_dataset.csv').exists():
        print("Creating improved high-quality dataset...")
        create_improved_dataset(
            stl_dir='STL',
            output_dir='improved_numpy_hq',
            dataset_csv='standardized_data/high_quality_dataset.csv'
        )
    
    # Process clean dataset if available
    if Path('standardized_data/clean_dataset.csv').exists():
        print("\nCreating improved clean dataset...")
        create_improved_dataset(
            stl_dir='STL',
            output_dir='improved_numpy_clean',
            dataset_csv='standardized_data/clean_dataset.csv'
        )
    else:
        print("No standardized datasets found. Run fix_data_issues.py first!")