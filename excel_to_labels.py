#!/usr/bin/env python3

import pandas as pd
import os
from pathlib import Path
import sys

def convert_excel_to_labels(excel_path, output_csv='classification_labels_new.csv'):
    """
    Convert Excel file with labels to CSV format for training
    
    Expected Excel format:
    - Column A: Model ID or filename (e.g., "3DModel1038604" or "3DModel1038604.stl")
    - Column B: Label (0, 1, or other integer values)
    
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
        
        if len(df.columns) < 2:
            print("Error: Excel file should have at least 2 columns (filename/ID and label)")
            return False
        
        # Look for filename and label columns by name or position
        filename_col = None
        label_col = None
        
        # Check for common column names
        for col in df.columns:
            col_lower = col.lower()
            if 'filename' in col_lower or 'file' in col_lower:
                filename_col = col
            elif 'label' in col_lower and label_col is None:
                label_col = col
            elif 'model_id' in col_lower and filename_col is None:
                filename_col = col
        
        # Fall back to positional if not found
        if filename_col is None:
            filename_col = df.columns[0]
        if label_col is None:
            # Try to find a numeric column that could be labels
            for col in df.columns[1:]:
                try:
                    pd.to_numeric(df[col])
                    label_col = col
                    break
                except:
                    continue
        
        if label_col is None:
            print("Error: Could not find a label column with numeric values")
            return False
        
        print(f"Using '{filename_col}' as filename column")
        print(f"Using '{label_col}' as label column")
        
        df_clean = pd.DataFrame()
        df_clean['filename'] = df[filename_col].astype(str)
        df_clean['label'] = pd.to_numeric(df[label_col], errors='coerce')
        
        df_clean['filename'] = df_clean['filename'].apply(lambda x: 
            x if x.endswith('.npy') else 
            (x.replace('.stl', '.npy') if '.stl' in x else f"{x}.npy"))
        
        df_clean = df_clean.dropna()
        
        df_clean['label'] = df_clean['label'].astype(int)
        
        numpy_dir = Path('numpy_arrays')
        existing_files = set(f.name for f in numpy_dir.glob('*.npy'))
        
        df_clean['exists'] = df_clean['filename'].apply(lambda x: x in existing_files)
        
        missing = df_clean[~df_clean['exists']]
        if len(missing) > 0:
            print(f"\nWarning: {len(missing)} files in Excel not found in numpy_arrays/:")
            for idx, row in missing.iterrows():
                print(f"  - {row['filename']}")
        
        found = df_clean[df_clean['exists']]
        print(f"\nMatched {len(found)} files with numpy arrays")
        
        final_df = found[['filename', 'label']]
        
        final_df.to_csv(output_csv, index=False)
        print(f"\nSaved {len(final_df)} labels to '{output_csv}'")
        
        label_counts = final_df['label'].value_counts().sort_index()
        print("\nLabel distribution:")
        for label, count in label_counts.items():
            print(f"  Class {label}: {count} samples ({count/len(final_df)*100:.1f}%)")
        
        orphaned_files = existing_files - set(final_df['filename'])
        if orphaned_files:
            print(f"\nNote: {len(orphaned_files)} numpy files without labels")
        
        return True
        
    except Exception as e:
        print(f"Error processing Excel file: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python excel_to_labels.py <excel_file> [output_csv]")
        print("\nExample:")
        print("  python excel_to_labels.py my_labels.xlsx")
        print("  python excel_to_labels.py my_labels.xlsx custom_labels.csv")
        sys.exit(1)
    
    excel_path = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else 'classification_labels_new.csv'
    
    success = convert_excel_to_labels(excel_path, output_csv)
    
    if success:
        print(f"\nConversion complete! Use the new labels for training:")
        print(f"  python train_subclavian.py --csv_file {output_csv}")
    else:
        print("\nConversion failed. Please check your Excel file format.")
        sys.exit(1)

if __name__ == "__main__":
    main()