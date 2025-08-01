#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from pathlib import Path
import random

def create_classification_labels(numpy_dir, output_csv="classification_labels.csv", seed=42):
    """
    Create a randomized CSV file with binary classification labels (0 and 1) 
    for all numpy point cloud files
    
    Args:
        numpy_dir: Directory containing .npy files
        output_csv: Output CSV filename
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    numpy_path = Path(numpy_dir)
    
    # Find all .npy files
    npy_files = list(numpy_path.glob("*.npy"))
    
    if len(npy_files) == 0:
        print(f"No .npy files found in {numpy_dir}")
        return
    
    print(f"Found {len(npy_files)} numpy files")
    
    # Create list to store data
    data = []
    
    for npy_file in sorted(npy_files):  # Sort for consistent ordering
        filename = npy_file.name
        # Generate random binary label (0 or 1)
        label = random.randint(0, 1)
        
        data.append({
            'filename': filename,
            'label': label
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Display distribution
    label_counts = df['label'].value_counts().sort_index()
    print(f"\nLabel distribution:")
    print(f"Class 0: {label_counts[0]} files")
    print(f"Class 1: {label_counts[1]} files")
    print(f"Balance: {label_counts[0]}/{label_counts[1]} = {label_counts[0]/label_counts[1]:.2f}")
    
    # Save to CSV
    output_path = numpy_path.parent / output_csv
    df.to_csv(output_path, index=False)
    
    print(f"\nClassification labels saved to: {output_path}")
    print(f"Total files: {len(df)}")
    
    # Display first few rows
    print(f"\nFirst 10 rows:")
    print(df.head(10))
    
    return df

if __name__ == "__main__":
    # Create labels for the numpy arrays
    numpy_directory = "numpy_arrays"
    create_classification_labels(numpy_directory)