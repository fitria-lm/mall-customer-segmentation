"""
Module untuk visualisasi data dan hasil clustering
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Import dari utils
from .utils import log_message

def set_plot_style(style='seaborn', context='notebook', palette='Set2'):
    """
    Set style untuk semua plot
    
    Args:
        style: Style matplotlib ('seaborn', 'ggplot', 'dark_background', dll)
        context: Seaborn context ('paper', 'notebook', 'talk', 'poster')
        palette: Color palette untuk seaborn
    """
    if style:
        plt.style.use(style)
    
    sns.set_context(context)
    sns.set_palette(palette)
    
    log_message(f"Plot style set to: {style}, context: {context}", "INFO")

def plot_elbow_method(k_values, wcss, optimal_k=None, save_path=None):
    """
    Plot elbow method untuk menentukan K optimal
    
    Args:
        k_values: List nilai K
        wcss: List WCSS untuk setiap K
        optimal_k: Nilai K optimal (jika diketahui)
        save_path: Path untuk menyimpan plot
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot elbow curve
    ax.plot(k_values, wcss, 'bo-', linewidth=2, markersize=8, label='WCSS')
    
    # Tandai optimal K jika diberikan
    if optimal_k and optimal_k in k_values:
        idx = k_values.index(optimal_k)
        ax.plot(k_values[idx], wcss[idx], 'ro', markersize=12, 
                label=f'Optimal K={optimal_k}')
        ax.axvline(x=optimal_k, color='r', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
    ax.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log_message(f"Elbow plot saved to: {save_path}", "SUCCESS")
    
    plt.show()
    
    return fig, ax

def plot_clusters_2d(df, x_col, y_col, cluster_col='Cluster', 
                    centers=None, save_path=None, figsize=(12, 8)):
    """
    Plot 2D clusters dengan dua fitur
    
    Args:
        df: DataFrame dengan data dan cluster labels
        x_col: Kolom untuk sumbu X
        y_col: Kolom untuk sumbu Y
        cluster_col: Kolom dengan cluster labels
        centers: Pusat cluster (optional)
        save_path: Path untuk menyimpan plot
        figsize: Ukuran figure
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot untuk setiap cluster
    clusters = df[cluster_col].unique()
    
    for cluster in sorted(clusters):
        cluster_data = df[df[cluster_col] == cluster]
        ax.scatter(
            cluster_data[x_col], 
            cluster_data[y_col],
            s=50,  # Marker size
            alpha=0.7,
            edgecolors='w',
            linewidth=0.5,
            label=f'Cluster {cluster}'
        )
    
    # Plot cluster centers jika diberikan
    if centers is not None:
        ax.scatter(
            centers[:, 0], 
            centers[:, 1],
            s=200,  # Larger markers for centers
            c='black',
            marker='X',
            label='Cluster Centers',
            edgecolors='w',
            linewidth=1.5
        )
    
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.set_title(f'Customer Segments: {x_col} vs {y_col}', 
                fontsize=14, fontweight='bold')
    ax.legend(title='Segments', loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log_message(f"2D cluster plot saved to: {save_path}", "SUCCESS")
    
    plt.show()
    
    return fig, ax

def plot_clusters_3d(df, x_col, y_col, z_col, cluster_col='Cluster', 
                    save_path=None, figsize=(14, 10)):
    """
    Plot 3D clusters
    
    Args:
        df: DataFrame dengan data dan cluster labels
        x_col: Kolom untuk sumbu X
        y_col: Kolom untuk sumbu Y
        z_col: Kolom untuk sumbu Z
        cluster_col: Kolom dengan cluster labels
        save_path: Path untuk menyimpan plot
        figsize: Ukuran figure
    """
    set_plot_style()
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot untuk setiap cluster
    clusters = df[cluster_col].unique()
    
    for cluster in sorted(clusters):
        cluster_data = df[df[cluster_col] == cluster]
        ax.scatter(
            cluster_data[x_col], 
            cluster_data[y_col],
            cluster_data[z_col],
            s=50,
            alpha=0.7,
            edgecolors='w',
            linewidth=0.5,
            label=f'Cluster {cluster}'
        )
    
    ax.set_xlabel(x_col, fontsize=11)
    ax.set_ylabel(y_col, fontsize=11)
    ax.set_zlabel(z_col, fontsize=11)
    ax.set_title(f'3D Customer Segments: {x_col} vs {y_col} vs {z_col}', 
                fontsize=14, fontweight='bold')
    ax.legend(title='Segments', loc='best')
    
    # Rotasi untuk view yang lebih baik
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log_message(f"3D cluster plot saved to: {save_path}", "SUCCESS")
    
    plt.show()
    
    return fig, ax

def plot_cluster_distribution(df, cluster_col='Cluster', save_path=None, figsize=(10, 6)):
    """
    Plot distribusi cluster (jumlah anggota per cluster)
    
    Args:
        df: DataFrame dengan cluster labels
        cluster_col: Kolom dengan cluster labels
        save_path: Path untuk menyimpan plot
        figsize: Ukuran figure
    """
    set_plot_style()
    
    # Hitung distribusi cluster
    cluster_counts = df[cluster_col].value_counts().sort_index()
    percentages = (cluster_counts / len(df)) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    bars = ax1.bar(range(len(cluster_counts)), cluster_counts.values, 
                  color=plt.cm.Set2(np.arange(len(cluster_counts)) / len(cluster_counts)))
    ax1.set_xlabel('Cluster', fontsize=12)
    ax1.set_ylabel('Number of Customers', fontsize=12)
    ax1.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(cluster_counts)))
    ax1.set_xticklabels([f'Cluster {i}' for i in cluster_counts.index])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Tambah nilai di atas bar
    for bar, count, pct in zip(bars, cluster_counts.values, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    # Pie chart
    wedges, texts, autotexts = ax2.pie(
        cluster_counts.values,
        labels=[f'Cluster {i}' for i in cluster_counts.index],
        autopct='%1.1f%%',
        startangle=90,
        colors=plt.cm.Set2(np.arange(len(cluster_counts)) / len(cluster_counts))
    )
    ax2.set_title('Cluster Proportion', fontsize=14, fontweight='bold')
    
    # Improve text visibility
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log_message(f"Cluster distribution plot saved to: {save_path}", "SUCCESS")
    
    plt.show()
    
    return fig, (ax1, ax2)

def plot_feature_boxplots(df, feature_cols, cluster_col='Cluster', 
                         save_path=None, figsize=(14, 10)):
    """
    Plot boxplot untuk setiap feature per cluster
    
    Args:
        df: DataFrame dengan data
        feature_cols: List features untuk diplot
        cluster_col: Kolom dengan cluster labels
        save_path: Path untuk menyimpan plot
        figsize: Ukuran figure
    """
    set_plot_style()
    
    n_features = len(feature_cols)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes array jika hanya 1 row/col
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, feature in enumerate(feature_cols):
        if idx < len(axes):
            # Buat boxplot untuk feature ini
            df.boxplot(column=feature, by=cluster_col, ax=axes[idx], grid=False)
            
            axes[idx].set_title(f'{feature} by Cluster', fontsize=12)
            axes[idx].set_xlabel('Cluster', fontsize=10)
            axes[idx].set_ylabel(feature, fontsize=10)
            
            # Remove automatic title yang dibuat oleh boxplot
            axes[idx].set_title(f'{feature} by Cluster', fontsize=12)
            fig.suptitle('')  # Remove automatic suptitle
    
    # Hide empty subplots
    for idx in range(len(feature_cols), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Feature Distribution by Cluster', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log_message(f"Feature boxplots saved to: {save_path}", "SUCCESS")
    
    plt.show()
    
    return fig, axes

# Test fungsi
if __name__ == "__main__":
    # Buat test data
    np.random.seed(42)
    n_samples = 100
    
    test_df = pd.DataFrame({
        'Feature1': np.random.randn(n_samples),
        'Feature2': np.random.randn(n_samples) * 2 + 5,
        'Feature3': np.random.randn(n_samples) * 0.5 + 10,
        'Cluster': np.random.randint(0, 3, n_samples)
    })
    
    print("Testing visualization functions...")
    
    # Test elbow plot (dummy data)
    k_values = list(range(2, 11))
    wcss = [1000 / (k**0.8) for k in k_values]  # Dummy WCSS values
    
    plot_elbow_method(k_values, wcss, optimal_k=5)
    
    # Test 2D cluster plot
    plot_clusters_2d(test_df, 'Feature1', 'Feature2', 'Cluster')
    
    print("✅ Test passed!")
