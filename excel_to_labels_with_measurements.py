#!/usr/bin/env python3

import pandas as pd
import os
from pathlib import Path
import sys

def convert_excel_with_measurements(excel_path, output_csv='classification_labels_with_measurements.csv'):
    """
    Convert Excel file with labels AND measurements to CSV format for training
    
    Extracts:
    - filename
    - label (binary classification)
    - left_subclavian_diameter_mm
    - aortic_arch_diameter_mm
    - angle_degrees
    
    Args:
        excel_path: Path to Excel file
        output_csv: Output CSV filename
    """
    
    if not os.path.exists(excel_path):
        print(f"Error: Excel file '{excel_path}' not found!")
        return False
    
    try:
        df = pd.read_excel(excel_path)
        print(f"Loaded Excel file with {len(df)} rows")
        print(f"Columns found: {df.columns.tolist()}")
        
        # Required columns
        required_cols = ['left_subclavian_diameter_mm', 'aortic_arch_diameter_mm', 'angle_degrees']
        
        # Find filename column
        filename_col = None
        for col in df.columns:
            col_lower = col.lower()
            if 'filename' in col_lower or 'file' in col_lower:
                filename_col = col
                break
            elif 'model_id' in col_lower:
                filename_col = col
        
        if filename_col is None:
            filename_col = df.columns[0]
        
        # Find label column
        label_col = None
        for col in df.columns:
            if 'label' in col.lower():
                label_col = col
                break
        
        if label_col is None:
            print("Error: Could not find label column")
            return False
        
        # Check for measurement columns
        missing_cols = []
        for col in required_cols:
            if col not in df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            print(f"Error: Missing required measurement columns: {missing_cols}")
            print("Required columns: left_subclavian_diameter_mm, aortic_arch_diameter_mm, angle_degrees")
            return False
        
        print(f"\nUsing columns:")
        print(f"  Filename: '{filename_col}'")
        print(f"  Label: '{label_col}'")
        print(f"  Measurements: {required_cols}")
        
        # Create clean dataframe
        df_clean = pd.DataFrame()
        
        # Process filename
        df_clean['filename'] = df[filename_col].astype(str)
        df_clean['filename'] = df_clean['filename'].apply(lambda x: 
            x if x.endswith('.npy') else 
            (x.replace('.stl', '.npy') if '.stl' in x else f"{x}.npy"))
        
        # Add label
        df_clean['label'] = pd.to_numeric(df[label_col], errors='coerce')
        
        # Add measurements
        for col in required_cols:
            df_clean[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN values
        initial_count = len(df_clean)
        df_clean = df_clean.dropna()
        if len(df_clean) < initial_count:
            print(f"\nRemoved {initial_count - len(df_clean)} rows with missing values")
        
        # Verify files exist
        numpy_dir = Path('numpy_arrays')
        existing_files = set(f.name for f in numpy_dir.glob('*.npy'))
        df_clean['exists'] = df_clean['filename'].apply(lambda x: x in existing_files)
        
        missing = df_clean[~df_clean['exists']]
        if len(missing) > 0:
            print(f"\nWarning: {len(missing)} files not found in numpy_arrays/")
            for idx, row in missing.head(5).iterrows():
                print(f"  - {row['filename']}")
            if len(missing) > 5:
                print(f"  ... and {len(missing)-5} more")
        
        # Keep only existing files
        final_df = df_clean[df_clean['exists']].drop('exists', axis=1)
        
        # Save to CSV
        final_df.to_csv(output_csv, index=False)
        print(f"\nSaved {len(final_df)} samples to '{output_csv}'")
        
        # Print statistics
        print("\nDataset Statistics:")
        print(f"  Total samples: {len(final_df)}")
        
        label_counts = final_df['label'].value_counts().sort_index()
        print("\nLabel distribution:")
        for label, count in label_counts.items():
            print(f"  Class {label}: {count} samples ({count/len(final_df)*100:.1f}%)")
        
        print("\nMeasurement ranges:")
        for col in required_cols:
            print(f"  {col}:")
            print(f"    Min: {final_df[col].min():.2f}")
            print(f"    Max: {final_df[col].max():.2f}")
            print(f"    Mean: {final_df[col].mean():.2f}")
            print(f"    Std: {final_df[col].std():.2f}")
        
        return True
        
    except Exception as e:
        print(f"Error processing Excel file: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python excel_to_labels_with_measurements.py <excel_file> [output_csv]")
        print("\nExample:")
        print("  python excel_to_labels_with_measurements.py labels/measurement_template1.xlsx")
        sys.exit(1)
    
    excel_path = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else 'classification_labels_with_measurements.csv'
    
    success = convert_excel_with_measurements(excel_path, output_csv)
    
    if success:
        print(f"\nConversion complete!")
        print(f"Next steps:")
        print(f"  1. Train with measurements: python train_with_measurements.py --csv_file {output_csv}")
        print(f"  2. Or use enhanced model: python pointnet_with_measurements.py --csv_file {output_csv}")
    else:
        print("\nConversion failed. Please check your Excel file format.")
        sys.exit(1)

if __name__ == "__main__":
    main()