#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

def create_measurement_template():
    """
    Create a template CSV file for you to fill in the measurements
    """
    # Read existing labels
    df = pd.read_csv('classification_labels.csv')
    
    # Add empty columns for measurements
    df['left_subclavian_diameter'] = np.nan  # in mm
    df['aortic_arch_diameter'] = np.nan      # in mm
    df['angle'] = np.nan                      # in degrees
    
    # Save template
    template_file = 'measurement_template.csv'
    df.to_csv(template_file, index=False)
    print(f"✅ Created template file: {template_file}")
    print(f"   Total files: {len(df)}")
    print("\n📝 Please fill in the measurement columns with your data:")
    print("   - left_subclavian_diameter (mm)")
    print("   - aortic_arch_diameter (mm)")
    print("   - angle (degrees)")
    
    return df

def integrate_measurements_from_excel(excel_file, sheet_name=None):
    """
    If you have measurements in Excel format
    
    Expected columns in Excel:
    - Model ID or filename
    - Left Subclavian Diameter
    - Aortic Arch Diameter  
    - Angle
    """
    print(f"Reading measurements from Excel: {excel_file}")
    
    # Read Excel file
    if sheet_name:
        df_measurements = pd.read_excel(excel_file, sheet_name=sheet_name)
    else:
        df_measurements = pd.read_excel(excel_file)
    
    print(f"Found {len(df_measurements)} rows in Excel")
    print(f"Columns: {df_measurements.columns.tolist()}")
    
    # Read existing labels
    df_labels = pd.read_csv('classification_labels.csv')
    
    # Show sample of Excel data for verification
    print("\nSample of Excel data (first 5 rows):")
    print(df_measurements.head())
    
    return df_measurements, df_labels

def integrate_measurements_manual(measurements_dict):
    """
    Manually integrate measurements from a dictionary
    
    Args:
        measurements_dict: Dictionary with format:
            {
                'model_id': {
                    'left_subclavian_diameter': value,
                    'aortic_arch_diameter': value,
                    'angle': value
                }
            }
    """
    df = pd.read_csv('classification_labels.csv')
    
    # Extract model ID from filename
    df['model_id'] = df['filename'].str.replace('.npy', '').str.replace('3DModel', '')
    
    # Add measurements
    for idx, row in df.iterrows():
        model_id = row['model_id']
        if model_id in measurements_dict:
            df.loc[idx, 'left_subclavian_diameter'] = measurements_dict[model_id]['left_subclavian_diameter']
            df.loc[idx, 'aortic_arch_diameter'] = measurements_dict[model_id]['aortic_arch_diameter']
            df.loc[idx, 'angle'] = measurements_dict[model_id]['angle']
    
    # Drop temporary model_id column
    df = df.drop('model_id', axis=1)
    
    return df

def validate_and_save_integrated_data(df, output_file='classification_labels_with_measurements.csv'):
    """
    Validate the integrated data and save
    """
    print("\n=== DATA VALIDATION ===")
    
    # Check for missing values
    missing_cols = ['left_subclavian_diameter', 'aortic_arch_diameter', 'angle']
    for col in missing_cols:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                print(f"⚠️  Missing {col}: {missing_count} files")
                # Show which files are missing
                missing_files = df[df[col].isna()]['filename'].tolist()[:5]
                print(f"   First few missing: {missing_files}")
    
    # Show statistics
    print("\n📊 Measurement Statistics:")
    if 'left_subclavian_diameter' in df.columns:
        print(f"Left Subclavian Diameter:")
        print(f"  Range: {df['left_subclavian_diameter'].min():.2f} - {df['left_subclavian_diameter'].max():.2f} mm")
        print(f"  Mean: {df['left_subclavian_diameter'].mean():.2f} mm")
        print(f"  Std: {df['left_subclavian_diameter'].std():.2f} mm")
    
    if 'aortic_arch_diameter' in df.columns:
        print(f"\nAortic Arch Diameter:")
        print(f"  Range: {df['aortic_arch_diameter'].min():.2f} - {df['aortic_arch_diameter'].max():.2f} mm")
        print(f"  Mean: {df['aortic_arch_diameter'].mean():.2f} mm")
        print(f"  Std: {df['aortic_arch_diameter'].std():.2f} mm")
    
    if 'angle' in df.columns:
        print(f"\nAngle:")
        print(f"  Range: {df['angle'].min():.2f} - {df['angle'].max():.2f} degrees")
        print(f"  Mean: {df['angle'].mean():.2f} degrees")
        print(f"  Std: {df['angle'].std():.2f} degrees")
    
    # Check correlation with labels
    print("\n📈 Correlation with Labels:")
    for col in missing_cols:
        if col in df.columns and not df[col].isna().all():
            correlation = df[['label', col]].corr().iloc[0, 1]
            print(f"  {col} vs label: {correlation:.3f}")
    
    # Save integrated data
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved integrated data to: {output_file}")
    
    # Create a version without missing values for immediate training
    df_complete = df.dropna(subset=missing_cols)
    if len(df_complete) < len(df):
        complete_file = output_file.replace('.csv', '_complete.csv')
        df_complete.to_csv(complete_file, index=False)
        print(f"✅ Saved complete data (no missing values) to: {complete_file}")
        print(f"   {len(df_complete)}/{len(df)} samples with complete measurements")
    
    return df

def fill_example_measurements():
    """
    Fill with example measurements for testing (replace with your real data)
    """
    df = pd.read_csv('classification_labels.csv')
    
    # Generate example measurements with some correlation to labels
    np.random.seed(42)
    n_samples = len(df)
    
    # Base measurements
    df['left_subclavian_diameter'] = np.random.normal(10, 2, n_samples)  # mean=10mm, std=2mm
    df['aortic_arch_diameter'] = np.random.normal(30, 5, n_samples)      # mean=30mm, std=5mm
    df['angle'] = np.random.normal(90, 20, n_samples)                    # mean=90°, std=20°
    
    # Add some correlation with labels (class 1 has different characteristics)
    class_1_mask = df['label'] == 1
    df.loc[class_1_mask, 'left_subclavian_diameter'] += np.random.normal(2, 0.5, class_1_mask.sum())
    df.loc[class_1_mask, 'aortic_arch_diameter'] += np.random.normal(3, 1, class_1_mask.sum())
    df.loc[class_1_mask, 'angle'] -= np.random.normal(10, 5, class_1_mask.sum())
    
    # Ensure positive values
    df['left_subclavian_diameter'] = df['left_subclavian_diameter'].clip(lower=5)
    df['aortic_arch_diameter'] = df['aortic_arch_diameter'].clip(lower=20)
    df['angle'] = df['angle'].clip(30, 150)
    
    return df

# Interactive integration helper
def interactive_integration():
    """
    Interactive helper to integrate your measurements
    """
    print("="*60)
    print("MEASUREMENT INTEGRATION HELPER")
    print("="*60)
    
    print("\nHow would you like to provide the measurements?")
    print("1. I have them in an Excel/CSV file")
    print("2. I'll fill in the template CSV manually")
    print("3. I have them in a different format")
    print("4. Generate example measurements for testing")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        file_path = input("Enter the path to your Excel/CSV file: ").strip()
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df_meas, df_labels = integrate_measurements_from_excel(file_path)
            # You'll need to map the columns here based on actual format
            print("\n⚠️  Please modify the integrate_measurements_from_excel function")
            print("    to map your Excel columns to the required format.")
        elif file_path.endswith('.csv'):
            df_meas = pd.read_csv(file_path)
            print(f"CSV columns: {df_meas.columns.tolist()}")
            print("\n⚠️  Please map these columns to the required format.")
            
    elif choice == '2':
        df_template = create_measurement_template()
        print("\n📋 Next steps:")
        print("1. Open 'measurement_template.csv'")
        print("2. Fill in the measurement columns")
        print("3. Save the file")
        print("4. Run: python integrate_measurements.py --validate measurement_template.csv")
        
    elif choice == '3':
        print("\n📝 Custom format integration:")
        print("Please modify the integrate_measurements_manual() function")
        print("to match your data format.")
        
    elif choice == '4':
        print("\n🔬 Generating example measurements for testing...")
        df = fill_example_measurements()
        df = validate_and_save_integrated_data(df)
        print("\n✅ Ready to train with example measurements!")
        print("Run: python train_with_measurements.py")
    
    else:
        print("Invalid choice. Please run again.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', type=str, help='Validate a filled measurement file')
    parser.add_argument('--example', action='store_true', help='Generate example measurements')
    parser.add_argument('--template', action='store_true', help='Create template CSV')
    
    args = parser.parse_args()
    
    if args.validate:
        df = pd.read_csv(args.validate)
        validate_and_save_integrated_data(df)
    elif args.example:
        df = fill_example_measurements()
        validate_and_save_integrated_data(df)
    elif args.template:
        create_measurement_template()
    else:
        interactive_integration()