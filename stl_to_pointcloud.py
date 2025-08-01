#!/usr/bin/env python3

import os
import glob
import trimesh
import numpy as np
from pathlib import Path
import argparse

def stl_to_pointcloud(stl_file, output_file, num_points=10000):
    """
    Convert STL file to point cloud file (.ply format)
    
    Args:
        stl_file: Path to input STL file
        output_file: Path to output point cloud file
        num_points: Number of points to sample from the mesh
    """
    try:
        # Load STL file
        mesh = trimesh.load(stl_file)
        
        # Sample points from the mesh surface
        points, face_indices = mesh.sample(num_points, return_index=True)
        
        # Get face normals for the sampled points
        face_normals = mesh.face_normals[face_indices]
        
        # Create point cloud with normals
        pointcloud = trimesh.PointCloud(vertices=points, vertex_normals=face_normals)
        
        # Export to PLY format
        pointcloud.export(output_file)
        
        return True
        
    except Exception as e:
        print(f"Error converting {stl_file}: {str(e)}")
        return False

def convert_all_stl_files(input_dir, output_dir=None, num_points=10000):
    """
    Convert all STL files in directory and subdirectories to point cloud files
    
    Args:
        input_dir: Directory containing STL files
        output_dir: Output directory (default: creates 'pointclouds' subdirectory)
        num_points: Number of points to sample from each mesh
    """
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_path = input_path / "pointclouds"
    else:
        output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True)
    
    # Find all STL files recursively
    stl_files = list(input_path.rglob("*.stl"))
    
    print(f"Found {len(stl_files)} STL files")
    
    converted_count = 0
    failed_count = 0
    
    for stl_file in stl_files:
        # Create relative path structure in output directory
        relative_path = stl_file.relative_to(input_path)
        output_file = output_path / relative_path.with_suffix('.ply')
        
        # Create subdirectories if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Converting: {stl_file.name} -> {output_file.name}")
        
        if stl_to_pointcloud(str(stl_file), str(output_file), num_points):
            converted_count += 1
        else:
            failed_count += 1
    
    print(f"\nConversion complete!")
    print(f"✅ Successfully converted: {converted_count} files")
    if failed_count > 0:
        print(f"❌ Failed conversions: {failed_count} files")

def main():
    parser = argparse.ArgumentParser(description='Convert STL files to point cloud files')
    parser.add_argument('input_dir', help='Input directory containing STL files')
    parser.add_argument('-o', '--output', help='Output directory (default: input_dir/pointclouds)')
    parser.add_argument('-n', '--num-points', type=int, default=10000, 
                       help='Number of points to sample (default: 10000)')
    
    args = parser.parse_args()
    
    convert_all_stl_files(args.input_dir, args.output, args.num_points)

if __name__ == "__main__":
    main()