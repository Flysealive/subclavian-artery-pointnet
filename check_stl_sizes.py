#!/usr/bin/env python3

import os
import trimesh
import numpy as np
from pathlib import Path
import pandas as pd

def check_stl_file_sizes(stl_dir):
    """
    Check file sizes and 3D dimensions of all STL files
    """
    stl_path = Path(stl_dir)
    stl_files = list(stl_path.glob("*.stl"))
    
    print(f"Found {len(stl_files)} STL files")
    
    results = []
    
    for stl_file in sorted(stl_files):
        try:
            # Get file size in bytes
            file_size = stl_file.stat().st_size
            file_size_mb = file_size / (1024 * 1024)  # Convert to MB
            
            # Load mesh and get 3D properties
            mesh = trimesh.load(str(stl_file))
            
            # Get bounding box dimensions
            bounds = mesh.bounds
            dimensions = bounds[1] - bounds[0]  # max - min
            width, height, depth = dimensions
            
            # Get mesh properties
            num_vertices = len(mesh.vertices)
            num_faces = len(mesh.faces)
            volume = mesh.volume if hasattr(mesh, 'volume') else 0
            surface_area = mesh.area if hasattr(mesh, 'area') else 0
            
            results.append({
                'filename': stl_file.name,
                'file_size_bytes': file_size,
                'file_size_mb': round(file_size_mb, 3),
                'width': round(width, 3),
                'height': round(height, 3),
                'depth': round(depth, 3),
                'num_vertices': num_vertices,
                'num_faces': num_faces,
                'volume': round(volume, 3),
                'surface_area': round(surface_area, 3)
            })
            
        except Exception as e:
            print(f"Error processing {stl_file.name}: {e}")
            results.append({
                'filename': stl_file.name,
                'file_size_bytes': stl_file.stat().st_size,
                'file_size_mb': round(stl_file.stat().st_size / (1024 * 1024), 3),
                'width': 'ERROR',
                'height': 'ERROR', 
                'depth': 'ERROR',
                'num_vertices': 'ERROR',
                'num_faces': 'ERROR',
                'volume': 'ERROR',
                'surface_area': 'ERROR'
            })
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv('stl_analysis.csv', index=False)
    
    print("\n=== FILE SIZE ANALYSIS ===")
    print(f"File size range: {df['file_size_mb'].min():.3f} MB - {df['file_size_mb'].max():.3f} MB")
    print(f"Average file size: {df['file_size_mb'].mean():.3f} MB")
    print(f"Standard deviation: {df['file_size_mb'].std():.3f} MB")
    
    # Check if all files are exactly the same size
    unique_sizes = df['file_size_bytes'].nunique()
    print(f"Number of unique file sizes: {unique_sizes}")
    
    if unique_sizes == 1:
        print("✅ All STL files are EXACTLY the same size!")
    else:
        print("❌ STL files have DIFFERENT sizes")
        
        # Show size distribution
        size_counts = df['file_size_mb'].value_counts().sort_index()
        print("\nFile size distribution:")
        for size, count in size_counts.items():
            print(f"  {size:.3f} MB: {count} files")
    
    print("\n=== 3D DIMENSIONS ANALYSIS ===")
    numeric_df = df[df['width'] != 'ERROR']
    if len(numeric_df) > 0:
        print(f"Width range: {numeric_df['width'].min():.3f} - {numeric_df['width'].max():.3f}")
        print(f"Height range: {numeric_df['height'].min():.3f} - {numeric_df['height'].max():.3f}")
        print(f"Depth range: {numeric_df['depth'].min():.3f} - {numeric_df['depth'].max():.3f}")
        
        print(f"\nAverage dimensions:")
        print(f"  Width: {numeric_df['width'].mean():.3f}")
        print(f"  Height: {numeric_df['height'].mean():.3f}")
        print(f"  Depth: {numeric_df['depth'].mean():.3f}")
    
    print("\n=== MESH COMPLEXITY ANALYSIS ===")
    if len(numeric_df) > 0:
        print(f"Vertices range: {numeric_df['num_vertices'].min()} - {numeric_df['num_vertices'].max()}")
        print(f"Faces range: {numeric_df['num_faces'].min()} - {numeric_df['num_faces'].max()}")
        print(f"Average vertices: {numeric_df['num_vertices'].mean():.0f}")
        print(f"Average faces: {numeric_df['num_faces'].mean():.0f}")
    
    print(f"\nDetailed analysis saved to: stl_analysis.csv")
    print(f"First 10 files:")
    print(df[['filename', 'file_size_mb', 'width', 'height', 'depth', 'num_vertices']].head(10))
    
    return df

if __name__ == "__main__":
    df = check_stl_file_sizes("STL")