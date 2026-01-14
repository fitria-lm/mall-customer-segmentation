import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd

def plot_elbow_method(k_range, wcss):
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.savefig('reports/figures/elbow_method.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_silhouette_scores(k_range, silhouette_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(k_range[1:], silhouette_scores, 'ro-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different k')
    plt.grid(True)
    plt.savefig('reports/figures/silhouette_scores.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_clusters_2d(df, labels, feature1, feature2):
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df[feature1], df[feature2], c=labels, cmap='viridis', alpha=0.6)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f'Customer Segments: {feature1} vs {feature2}')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'reports/figures/clusters_{feature1}_{feature2}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_clusters_3d(df, labels):
    fig = px.scatter_3d(
        df, 
        x='Annual Income (k$)', 
        y='Spending Score (1-100)', 
        z='Age',
        color=labels.astype(str),
        title='3D Customer Segments',
        labels={'color': 'Cluster'},
        opacity=0.7
    )
    fig.write_html('reports/figures/clusters_3d.html')
    fig.show()

def plot_cluster_profiles(df_original, labels, numerical_features):
    df_analysis = df_original.copy()
    df_analysis['Cluster'] = labels
    
    cluster_profiles = df_analysis.groupby('Cluster')[numerical_features].mean().round(2)
    cluster_sizes = df_analysis['Cluster'].value_counts().sort_index()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, feature in enumerate(numerical_features):
        axes[i].bar(cluster_profiles.index, cluster_profiles[feature])
        axes[i].set_xlabel('Cluster')
        axes[i].set_ylabel(f'Average {feature}')
        axes[i].set_title(f'Average {feature} by Cluster')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/cluster_profiles.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cluster_profiles, cluster_sizes