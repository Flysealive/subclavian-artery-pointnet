import numpy as np
import trimesh
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def stl_to_voxel(stl_path, voxel_size=64):
    """
    Convert STL file to voxel representation
    
    Args:
        stl_path: Path to STL file
        voxel_size: Size of voxel grid (default 64x64x64)
    
    Returns:
        voxel_grid: 3D numpy array of shape (voxel_size, voxel_size, voxel_size)
    """
    mesh = trimesh.load(stl_path)
    
    if not mesh.is_watertight:
        mesh.fill_holes()
    
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    scale = (bounds[1] - bounds[0]).max()
    
    mesh.vertices = (mesh.vertices - center) / scale * (voxel_size - 1)
    mesh.vertices += voxel_size / 2
    
    voxel_grid = mesh.voxelized(pitch=1.0).matrix
    
    # Ensure output is exactly voxel_size x voxel_size x voxel_size
    output = np.zeros((voxel_size, voxel_size, voxel_size), dtype=np.float32)
    
    # Get the actual size of the voxel grid
    grid_shape = voxel_grid.shape
    
    # Calculate start positions for centering
    start_x = max(0, (voxel_size - grid_shape[0]) // 2)
    start_y = max(0, (voxel_size - grid_shape[1]) // 2)
    start_z = max(0, (voxel_size - grid_shape[2]) // 2)
    
    # Calculate end positions
    end_x = min(start_x + grid_shape[0], voxel_size)
    end_y = min(start_y + grid_shape[1], voxel_size)
    end_z = min(start_z + grid_shape[2], voxel_size)
    
    # Calculate source slicing if grid is larger than target
    src_start_x = max(0, (grid_shape[0] - voxel_size) // 2)
    src_start_y = max(0, (grid_shape[1] - voxel_size) // 2)
    src_start_z = max(0, (grid_shape[2] - voxel_size) // 2)
    
    src_end_x = min(src_start_x + voxel_size, grid_shape[0])
    src_end_y = min(src_start_y + voxel_size, grid_shape[1])
    src_end_z = min(src_start_z + voxel_size, grid_shape[2])
    
    # Copy the voxel data
    output[start_x:end_x, start_y:end_y, start_z:end_z] = \
        voxel_grid[src_start_x:src_end_x, src_start_y:src_end_y, src_start_z:src_end_z]
    
    return output

def convert_dataset_to_voxels(stl_dir='STL', output_dir='voxel_arrays', voxel_size=64):
    """
    Convert all STL files in directory to voxel representations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    stl_files = [f for f in os.listdir(stl_dir) if f.endswith('.stl')]
    
    print(f"Converting {len(stl_files)} STL files to voxels...")
    
    for stl_file in tqdm(stl_files):
        stl_path = os.path.join(stl_dir, stl_file)
        output_path = os.path.join(output_dir, stl_file.replace('.stl', '.npy'))
        
        try:
            voxel_grid = stl_to_voxel(stl_path, voxel_size)
            np.save(output_path, voxel_grid)
        except Exception as e:
            print(f"Error processing {stl_file}: {e}")
            continue
    
    print(f"Conversion complete! Voxel arrays saved to {output_dir}/")

if __name__ == "__main__":
    convert_dataset_to_voxels(voxel_size=64)