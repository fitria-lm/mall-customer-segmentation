import pandas as pd
import numpy as np
import yaml
import joblib
import os
from src.data_loader import load_data, load_config
from src.preprocessing import preprocess_data
from src.clustering import find_optimal_k, apply_kmeans
from src.visualization import (
    plot_elbow_method, plot_silhouette_scores,
    plot_clusters_2d, plot_clusters_3d,
    plot_cluster_profiles
)

def create_directories():
    directories = ['data/processed', 'models', 'reports/figures', 'notebooks']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    create_directories()
    
    config = load_config()
    
    print("Step 1: Loading data...")
    df = load_data()
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("\nStep 2: Data preprocessing...")
    df_processed, scaler = preprocess_data(df, config)
    
    print("\nStep 3: Finding optimal number of clusters...")
    optimal_k, wcss, silhouette_scores = find_optimal_k(
        df_processed[config['features']['numerical']], 
        config['clustering']['k_range']
    )
    print(f"Optimal number of clusters: {optimal_k}")
    
    plot_elbow_method(config['clustering']['k_range'], wcss)
    plot_silhouette_scores(config['clustering']['k_range'], silhouette_scores)
    
    print("\nStep 4: Applying K-Means clustering...")
    kmeans, labels = apply_kmeans(
        df_processed[config['features']['numerical']], 
        optimal_k
    )
    
    joblib.dump(kmeans, f"{config['paths']['models']}/kmeans_model.pkl")
    
    print("\nStep 5: Visualizing clusters...")
    plot_clusters_2d(
        df_processed, 
        labels, 
        'Annual Income (k$)', 
        'Spending Score (1-100)'
    )
    
    plot_clusters_2d(
        df_processed, 
        labels, 
        'Age', 
        'Spending Score (1-100)'
    )
    
    try:
        plot_clusters_3d(df_processed, labels)
    except:
        print("3D plot skipped (plotly not available)")
    
    print("\nStep 6: Analyzing cluster profiles...")
    cluster_profiles, cluster_sizes = plot_cluster_profiles(
        df, 
        labels, 
        config['features']['numerical']
    )
    
    print("\nCluster Profiles:")
    print(cluster_profiles)
    print("\nCluster Sizes:")
    print(cluster_sizes)
    
    print("\nStep 7: Saving results...")
    df_result = df.copy()
    df_result['Cluster'] = labels
    df_result.to_csv(config['data']['processed_path'], index=False)
    
    print(f"\nResults saved to: {config['data']['processed_path']}")
    print("Project completed successfully!")

if __name__ == "__main__":
    main()