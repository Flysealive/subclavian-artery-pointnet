#!/usr/bin/env python3

import os
import numpy as np
import trimesh
from pathlib import Path
import argparse

def ply_to_numpy(ply_file, output_file):
    """
    Convert PLY point cloud file to numpy array file (.npy format)
    
    Args:
        ply_file: Path to input PLY file
        output_file: Path to output numpy file
    """
    try:
        # Load PLY file
        pointcloud = trimesh.load(ply_file)
        
        # Extract vertices (points) and normals if available
        points = pointcloud.vertices
        
        # Check if normals are available
        if hasattr(pointcloud, 'vertex_normals') and pointcloud.vertex_normals is not None:
            # Combine points and normals (Nx6 array: x,y,z,nx,ny,nz)
            data = np.hstack([points, pointcloud.vertex_normals])
        else:
            # Only points (Nx3 array: x,y,z)
            data = points
        
        # Save as numpy array
        np.save(output_file, data)
        
        return True, data.shape
        
    except Exception as e:
        print(f"Error converting {ply_file}: {str(e)}")
        return False, None

def convert_all_ply_files(input_dir, output_dir=None):
    """
    Convert all PLY files in directory to numpy array files
    
    Args:
        input_dir: Directory containing PLY files
        output_dir: Output directory (default: creates 'numpy_arrays' subdirectory)
    """
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_path = input_path.parent / "numpy_arrays"
    else:
        output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True)
    
    # Find all PLY files
    ply_files = list(input_path.glob("*.ply"))
    
    print(f"Found {len(ply_files)} PLY files")
    
    converted_count = 0
    failed_count = 0
    
    for ply_file in ply_files:
        # Create output filename
        output_file = output_path / ply_file.with_suffix('.npy').name
        
        print(f"Converting: {ply_file.name} -> {output_file.name}")
        
        success, shape = ply_to_numpy(str(ply_file), str(output_file))
        if success:
            converted_count += 1
            print(f"  Shape: {shape}")
        else:
            failed_count += 1
    
    print(f"\nConversion complete!")
    print(f"✅ Successfully converted: {converted_count} files")
    if failed_count > 0:
        print(f"❌ Failed conversions: {failed_count} files")

def main():
    parser = argparse.ArgumentParser(description='Convert PLY point cloud files to numpy arrays')
    parser.add_argument('input_dir', help='Input directory containing PLY files')
    parser.add_argument('-o', '--output', help='Output directory (default: input_dir/../numpy_arrays)')
    
    args = parser.parse_args()
    
    convert_all_ply_files(args.input_dir, args.output)

if __name__ == "__main__":
    main()