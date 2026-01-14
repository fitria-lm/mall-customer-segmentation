"""
Main script untuk Mall Customer Segmentation
"""

# Import standar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import dari package src (karena ada __init__.py)
from src import (
    load_data, 
    preprocess_data, 
    kmeans_clustering, 
    find_optimal_k,
    plot_clusters_2d,
    plot_elbow_method,
    plot_cluster_distribution,
    plot_feature_boxplots
)

# Import utils untuk setup
from src.utils import setup_project, log_message, save_results, save_model_artifacts

# ===== KONFIGURASI =====
CONFIG = {
    'data_path': 'data/raw/Mall_Customers.csv',
    'processed_path': 'data/processed/customers_clustered.csv',
    'optimal_k_method': 'silhouette',  # 'elbow' atau 'silhouette'
    'max_k': 10,
    'random_state': 42,
    'save_model': True,
    'create_plots': True
}

# ===== FUNGSI UTAMA =====
def run_analysis(config):
    """
    Jalankan seluruh pipeline analisis
    """
    log_message("Starting Mall Customer Segmentation Analysis", "INFO")
    log_message(f"Configuration: {config}", "INFO")
    
    # Step 1: Setup project structure
    setup_project()
    
    # Step 2: Load data
    log_message("=" * 50, "INFO")
    log_message("STEP 1: LOADING DATA", "INFO")
    df = load_data(config['data_path'])
    
    # Display basic info
    print("\n" + "="*50)
    print("DATA OVERVIEW")
    print("="*50)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Step 3: Preprocess data
    log_message("=" * 50, "INFO")
    log_message("STEP 2: PREPROCESSING DATA", "INFO")
    df_processed, X_scaled, scaler, features = preprocess_data(df)
    
    print(f"\nFeatures selected for clustering: {features}")
    print(f"Scaled data shape: {X_scaled.shape}")
    
    # Step 4: Find optimal K
    log_message("=" * 50, "INFO")
    log_message("STEP 3: FINDING OPTIMAL NUMBER OF CLUSTERS", "INFO")
    
    k_results = find_optimal_k(
        X_scaled, 
        max_k=config['max_k'], 
        method=config['optimal_k_method']
    )
    
    optimal_k = k_results['optimal_k']
    print(f"\nOptimal number of clusters: {optimal_k}")
    print(f"Method used: {config['optimal_k_method']}")
    
    # Plot elbow method
    if config['create_plots']:
        plot_elbow_method(
            k_results['k_values'], 
            k_results['wcss'], 
            optimal_k=optimal_k,
            save_path='reports/figures/elbow_method.png'
        )
    
    # Step 5: Apply K-Means with optimal K
    log_message("=" * 50, "INFO")
    log_message(f"STEP 4: APPLYING K-MEANS WITH K={optimal_k}", "INFO")
    
    labels, kmeans_model = kmeans_clustering(
        X_scaled, 
        n_clusters=optimal_k,
        random_state=config['random_state']
    )
    
    # Add cluster labels to processed dataframe
    df_processed['Cluster'] = labels
    
    # Step 6: Analyze clusters
    log_message("=" * 50, "INFO")
    log_message("STEP 5: ANALYZING CLUSTERS", "INFO")
    
    # Basic cluster statistics
    print("\n" + "="*50)
    print("CLUSTER DISTRIBUTION")
    print("="*50)
    cluster_dist = df_processed['Cluster'].value_counts().sort_index()
    for cluster, count in cluster_dist.items():
        percentage = (count / len(df_processed)) * 100
        print(f"Cluster {cluster}: {count} customers ({percentage:.1f}%)")
    
    # Step 7: Visualize results
    if config['create_plots']:
        log_message("=" * 50, "INFO")
        log_message("STEP 6: VISUALIZING RESULTS", "INFO")
        
        # Plot 2D: Income vs Spending Score
        plot_clusters_2d(
            df_processed,
            x_col='Annual Income (k$)',
            y_col='Spending Score (1-100)',
            cluster_col='Cluster',
            save_path='reports/figures/clusters_2d_income_spending.png'
        )
        
        # Plot 2D: Age vs Spending Score
        plot_clusters_2d(
            df_processed,
            x_col='Age',
            y_col='Spending Score (1-100)',
            cluster_col='Cluster',
            save_path='reports/figures/clusters_2d_age_spending.png'
        )
        
        # Plot cluster distribution
        plot_cluster_distribution(
            df_processed,
            cluster_col='Cluster',
            save_path='reports/figures/cluster_distribution.png'
        )
        
        # Plot feature boxplots
        plot_feature_boxplots(
            df_processed,
            feature_cols=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
            cluster_col='Cluster',
            save_path='reports/figures/feature_boxplots.png'
        )
    
    # Step 8: Save results
    log_message("=" * 50, "INFO")
    log_message("STEP 7: SAVING RESULTS", "INFO")
    
    # Save processed data with clusters
    output_path = save_results(
        df_processed, 
        config['processed_path'],
        include_timestamp=True
    )
    
    # Save model artifacts if configured
    if config['save_model']:
        save_model_artifacts(kmeans_model, scaler, filename_prefix="mall_segmentation")
    
    # Step 9: Generate summary report
    log_message("=" * 50, "INFO")
    log_message("STEP 8: GENERATING SUMMARY", "INFO")
    
    generate_summary_report(df_processed, optimal_k, output_path)
    
    log_message("=" * 50, "INFO")
    log_message("ANALYSIS COMPLETED SUCCESSFULLY!", "SUCCESS")
    log_message("=" * 50, "INFO")
    
    return df_processed, kmeans_model, scaler

def generate_summary_report(df, optimal_k, output_path):
    """
    Generate summary report dari analisis
    """
    report = f"""
    ===========================================
    MALL CUSTOMER SEGMENTATION - SUMMARY REPORT
    ===========================================
    
    Analysis Date: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    1. DATA OVERVIEW
       --------------
       Total Customers: {len(df)}
       Features Used: 5 (CustomerID, Genre, Age, Annual Income, Spending Score)
    
    2. CLUSTERING RESULTS
       ------------------
       Optimal Clusters: {optimal_k}
       Clustering Algorithm: K-Means (K-Means++ initialization)
    
    3. CLUSTER DISTRIBUTION
       ---------------------
    """
    
    # Add cluster statistics
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        size = len(cluster_data)
        percentage = (size / len(df)) * 100
        
        # Calculate averages
        avg_age = cluster_data['Age'].mean()
        avg_income = cluster_data['Annual Income (k$)'].mean()
        avg_spending = cluster_data['Spending Score (1-100)'].mean()
        
        # Determine cluster characteristics
        age_desc = "Young" if avg_age < 30 else "Middle-aged" if avg_age < 50 else "Senior"
        income_desc = "Low" if avg_income < 40 else "Medium" if avg_income < 70 else "High"
        spending_desc = "Low" if avg_spending < 40 else "Medium" if avg_spending < 60 else "High"
        
        report += f"""
       Cluster {cluster}:
         • Size: {size} customers ({percentage:.1f}%)
         • Average Age: {avg_age:.1f} years ({age_desc})
         • Average Income: ${avg_income:.1f}k ({income_desc})
         • Average Spending Score: {avg_spending:.1f}/100 ({spending_desc})
         • Profile: {age_desc}, {income_desc} income, {spending_desc} spending
        """
    
    report += f"""
    4. OUTPUT
       -------
       Processed Data: {output_path}
       Visualizations: reports/figures/
       Model Artifacts: models/ (if saved)
    
    5. RECOMMENDATIONS
       ----------------
       1. Target Cluster dengan spending score tinggi untuk promosi premium
       2. Cluster dengan income tinggi tapi spending rendah perlu strategi khusus
       3. Personalisasi marketing campaign berdasarkan karakteristik cluster
    
    ===========================================
    END OF REPORT
    ===========================================
    """
    
    # Save report to file
    report_path = 'reports/summary_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    log_message(f"Summary report saved to: {report_path}", "SUCCESS")
    
    # Print summary to console
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total customers analyzed: {len(df)}")
    print(f"Optimal clusters found: {optimal_k}")
    print(f"Results saved to: {output_path}")
    print("="*60)

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    try:
        # Run the analysis
        df_final, model, scaler = run_analysis(CONFIG)
        
        # Optional: Show final dataframe info
        print("\n" + "="*60)
        print("FINAL DATA PREVIEW (with clusters)")
        print("="*60)
        print(df_final[['Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']].head(10))
        
    except Exception as e:
        log_message(f"Error during analysis: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
