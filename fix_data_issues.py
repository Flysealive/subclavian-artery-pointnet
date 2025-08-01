#!/usr/bin/env python3

import pandas as pd
import numpy as np
import trimesh
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_outliers_and_quality(csv_file='stl_analysis.csv'):
    """
    Identify outliers and quality issues in the STL dataset
    """
    df = pd.read_csv(csv_file)
    
    print("=== DATA QUALITY ANALYSIS ===\n")
    
    # 1. Identify outliers using IQR method
    def find_outliers(column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers, lower_bound, upper_bound
    
    # Check for outliers in key metrics
    metrics = ['file_size_mb', 'num_vertices', 'num_faces', 'volume', 'surface_area']
    outlier_summary = {}
    
    for metric in metrics:
        outliers, lower, upper = find_outliers(metric)
        outlier_summary[metric] = {
            'count': len(outliers),
            'files': outliers['filename'].tolist() if len(outliers) > 0 else [],
            'range': f"{lower:.3f} - {upper:.3f}"
        }
        
        print(f"{metric.upper()} OUTLIERS:")
        print(f"  Normal range: {lower:.3f} - {upper:.3f}")
        print(f"  Outliers found: {len(outliers)}")
        if len(outliers) > 0:
            print(f"  Files: {outliers['filename'].tolist()[:5]}{'...' if len(outliers) > 5 else ''}")
        print()
    
    # 2. Identify extremely small/large models
    print("=== SIZE CATEGORIES ===")
    
    # Categorize by file size
    df['size_category'] = pd.cut(df['file_size_mb'], 
                                bins=[0, 1, 2, 3, 5], 
                                labels=['Small (<1MB)', 'Medium (1-2MB)', 'Large (2-3MB)', 'XLarge (>3MB)'])
    
    size_dist = df['size_category'].value_counts()
    print("File size distribution:")
    for category, count in size_dist.items():
        print(f"  {category}: {count} files")
    print()
    
    # 3. Identify potential quality issues
    print("=== POTENTIAL QUALITY ISSUES ===")
    
    # Very low complexity models (might be incomplete/simplified)
    low_complexity = df[df['num_vertices'] < 10000]
    print(f"Low complexity models (<10k vertices): {len(low_complexity)}")
    if len(low_complexity) > 0:
        print(f"  Files: {low_complexity['filename'].tolist()[:5]}")
    
    # Very high complexity models (might have scanning artifacts)
    high_complexity = df[df['num_vertices'] > 35000]
    print(f"High complexity models (>35k vertices): {len(high_complexity)}")
    if len(high_complexity) > 0:
        print(f"  Files: {high_complexity['filename'].tolist()[:5]}")
    
    # Unusual aspect ratios
    df['aspect_ratio'] = df['width'] / df['height']
    unusual_aspect = df[(df['aspect_ratio'] < 0.3) | (df['aspect_ratio'] > 2.0)]
    print(f"Unusual aspect ratios: {len(unusual_aspect)}")
    if len(unusual_aspect) > 0:
        print(f"  Files: {unusual_aspect['filename'].tolist()[:5]}")
    
    return df, outlier_summary

def create_standardized_dataset(df, output_dir='standardized_data'):
    """
    Create a standardized version of the dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\n=== CREATING STANDARDIZED DATASET ===")
    
    # Strategy 1: Remove extreme outliers
    def remove_outliers(df, columns, threshold=2.5):
        """Remove samples that are outliers in multiple metrics"""
        outlier_counts = pd.Series(index=df.index, data=0)
        
        for col in columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_counts += (z_scores > threshold).astype(int)
        
        # Keep samples that are outliers in fewer than 2 metrics
        clean_df = df[outlier_counts < 2].copy()
        removed_df = df[outlier_counts >= 2].copy()
        
        return clean_df, removed_df
    
    # Remove extreme outliers
    metrics_to_check = ['file_size_mb', 'num_vertices', 'volume']
    clean_df, removed_df = remove_outliers(df, metrics_to_check)
    
    print(f"Original dataset: {len(df)} files")
    print(f"Clean dataset: {len(clean_df)} files")
    print(f"Removed outliers: {len(removed_df)} files")
    
    if len(removed_df) > 0:
        print(f"Removed files: {removed_df['filename'].tolist()}")
        removed_df.to_csv(output_path / 'removed_outliers.csv', index=False)
    
    # Strategy 2: Create size-balanced subsets
    print(f"\n=== SIZE-BALANCED SUBSETS ===")
    
    # Create balanced dataset by size categories
    size_categories = clean_df['size_category'].value_counts()
    min_samples = min(size_categories.values)
    
    balanced_df = pd.DataFrame()
    for category in size_categories.index:
        category_samples = clean_df[clean_df['size_category'] == category].sample(n=min_samples, random_state=42)
        balanced_df = pd.concat([balanced_df, category_samples])
    
    print(f"Balanced dataset: {len(balanced_df)} files ({min_samples} per size category)")
    
    # Save datasets
    clean_df.to_csv(output_path / 'clean_dataset.csv', index=False)
    balanced_df.to_csv(output_path / 'balanced_dataset.csv', index=False)
    
    # Strategy 3: Quality-based filtering
    print(f"\n=== QUALITY-BASED FILTERING ===")
    
    # High-quality subset (good complexity, reasonable size)
    quality_df = clean_df[
        (clean_df['num_vertices'] >= 12000) &  # Sufficient detail
        (clean_df['num_vertices'] <= 30000) &  # Not too complex
        (clean_df['file_size_mb'] >= 1.0) &    # Not too small
        (clean_df['file_size_mb'] <= 3.0) &    # Not too large
        (clean_df['aspect_ratio'] >= 0.4) &    # Reasonable proportions
        (clean_df['aspect_ratio'] <= 1.5)
    ].copy()
    
    print(f"High-quality subset: {len(quality_df)} files")
    quality_df.to_csv(output_path / 'high_quality_dataset.csv', index=False)
    
    return {
        'clean': clean_df,
        'balanced': balanced_df,
        'high_quality': quality_df,
        'removed': removed_df
    }

def generate_improved_labels(datasets, output_dir='standardized_data'):
    """
    Generate more meaningful labels based on anatomical features
    """
    output_path = Path(output_dir)
    
    print(f"\n=== GENERATING IMPROVED LABELS ===")
    
    for name, df in datasets.items():
        if name == 'removed' or len(df) == 0:
            continue
            
        print(f"\nCreating labels for {name} dataset ({len(df)} files):")
        
        # Strategy 1: Size-based classification
        df_copy = df.copy()
        df_copy['size_label'] = (df_copy['file_size_mb'] > df_copy['file_size_mb'].median()).astype(int)
        size_dist = df_copy['size_label'].value_counts()
        print(f"  Size-based: {size_dist[0]} small, {size_dist[1]} large")
        
        # Strategy 2: Complexity-based classification  
        df_copy['complexity_label'] = (df_copy['num_vertices'] > df_copy['num_vertices'].median()).astype(int)
        complexity_dist = df_copy['complexity_label'].value_counts()
        print(f"  Complexity-based: {complexity_dist[0]} simple, {complexity_dist[1]} complex")
        
        # Strategy 3: Volume-based classification
        df_copy['volume_label'] = (df_copy['volume'] > df_copy['volume'].median()).astype(int)
        volume_dist = df_copy['volume_label'].value_counts()
        print(f"  Volume-based: {volume_dist[0]} small volume, {volume_dist[1]} large volume")
        
        # Strategy 4: Multi-feature clustering (binary)
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        
        features = ['file_size_mb', 'num_vertices', 'volume', 'surface_area']
        feature_data = df_copy[features].values
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        
        kmeans = KMeans(n_clusters=2, random_state=42)
        df_copy['cluster_label'] = kmeans.fit_predict(scaled_features)
        cluster_dist = df_copy['cluster_label'].value_counts()
        print(f"  Cluster-based: {cluster_dist[0]} cluster A, {cluster_dist[1]} cluster B")
        
        # Save labels
        label_columns = ['filename', 'size_label', 'complexity_label', 'volume_label', 'cluster_label']
        df_copy[label_columns].to_csv(output_path / f'{name}_improved_labels.csv', index=False)

if __name__ == "__main__":
    # Run analysis
    df, outliers = analyze_outliers_and_quality()
    
    # Create standardized datasets
    datasets = create_standardized_dataset(df)
    
    # Generate improved labels
    generate_improved_labels(datasets)
    
    print(f"\n=== SUMMARY ===")
    print(f"✅ Analysis complete!")
    print(f"✅ Standardized datasets created in 'standardized_data/' folder")
    print(f"✅ Improved labels generated for meaningful classification")
    print(f"\nRecommendation: Use 'high_quality_dataset.csv' with 'cluster_label' for best results!")