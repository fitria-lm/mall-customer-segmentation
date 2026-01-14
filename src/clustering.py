"""
Module untuk clustering dengan K-Means
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Import dari utils
from .utils import log_message, print_progress

def find_optimal_k(X, max_k=10, method='elbow'):
    """
    Cari jumlah cluster optimal
    
    Args:
        X: Data untuk clustering
        max_k: Maximum K untuk dicoba
        method: 'elbow' atau 'silhouette'
    
    Returns:
        dict: Hasil evaluasi untuk setiap K
    """
    log_message(f"Finding optimal K (max={max_k}, method={method})", "INFO")
    
    wcss = []  # Within-cluster sum of squares
    silhouette_scores = []
    models = []
    
    k_range = range(2, max_k + 1)
    
    for i, k in enumerate(k_range):
        print_progress(i, len(k_range), prefix='Testing K:', suffix='Complete')
        
        # Buat dan fit model
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X)
        
        # Simpan model
        models.append(kmeans)
        
        # Hitung WCSS
        wcss.append(kmeans.inertia_)
        
        # Hitung silhouette score (jika lebih dari 1 cluster)
        if k > 1:
            score = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(0)
    
    print_progress(len(k_range), len(k_range), prefix='Testing K:', suffix='Complete')
    
    # Tentukan K optimal berdasarkan method
    optimal_k = None
    
    if method == 'elbow':
        # Cari "elbow point" - di mana penurunan WCSS mulai melandai
        diffs = np.diff(wcss)
        diff_ratios = np.diff(diffs) / diffs[:-1]
        optimal_k = k_range[np.argmin(diff_ratios) + 1]
        
    elif method == 'silhouette':
        # Pilih K dengan silhouette score tertinggi
        optimal_k = k_range[np.argmax(silhouette_scores)]
    
    results = {
        'k_values': list(k_range),
        'wcss': wcss,
        'silhouette_scores': silhouette_scores,
        'models': models,
        'optimal_k': optimal_k,
        'optimal_k_method': method
    }
    
    log_message(f"Optimal K found: {optimal_k} (using {method} method)", "SUCCESS")
    
    return results

def kmeans_clustering(X, n_clusters=5, random_state=42, return_model=True):
    """
    Lakukan K-Means clustering
    
    Args:
        X: Data untuk clustering
        n_clusters: Jumlah cluster
        random_state: Random seed untuk reproducibility
        return_model: Return model bersama labels
    
    Returns:
        tuple atau array: (labels, model) atau labels saja
    """
    log_message(f"Performing K-Means clustering with {n_clusters} clusters", "INFO")
    
    # Buat dan fit model
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        random_state=random_state,
        n_init=10
    )
    
    labels = kmeans.fit_predict(X)
    
    # Hitung metrics
    wcss = kmeans.inertia_
    if n_clusters > 1:
        silhouette = silhouette_score(X, labels)
        log_message(f"WCSS: {wcss:.2f}, Silhouette Score: {silhouette:.4f}", "INFO")
    else:
        log_message(f"WCSS: {wcss:.2f}", "INFO")
    
    log_message(f"Clustering complete. Cluster distribution:", "SUCCESS")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        percentage = (count / len(labels)) * 100
        print(f"  Cluster {cluster}: {count} samples ({percentage:.1f}%)")
    
    if return_model:
        return labels, kmeans
    else:
        return labels

def analyze_clusters(df, labels, feature_names=None):
    """
    Analisis karakteristik setiap cluster
    
    Args:
        df: DataFrame original
        labels: Cluster labels
        feature_names: Features untuk dianalisis
    
    Returns:
        DataFrame: Statistik per cluster
    """
    log_message("Analyzing cluster characteristics", "INFO")
    
    # Tambah labels ke DataFrame
    df_analyze = df.copy()
    df_analyze['Cluster'] = labels
    
    # Jika feature_names tidak ditentukan, gunakan semua numeric columns
    if feature_names is None:
        feature_names = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Cluster' in feature_names:
            feature_names.remove('Cluster')
    
    # Hitung statistik per cluster
    cluster_stats = df_analyze.groupby('Cluster')[feature_names].agg(['mean', 'std', 'count'])
    
    # Flatten multi-index columns
    cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
    
    # Tambah persentase
    total_samples = len(df_analyze)
    count_cols = [col for col in cluster_stats.columns if 'count' in col]
    if count_cols:
        cluster_stats['percentage'] = (cluster_stats[count_cols[0]] / total_samples * 100)
    
    log_message(f"Cluster analysis complete for {len(cluster_stats)} clusters", "SUCCESS")
    
    return cluster_stats, df_analyze

# Test fungsi
if __name__ == "__main__":
    # Generate test data
    np.random.seed(42)
    X_test = np.random.randn(100, 3)
    
    print("Testing clustering functions...")
    
    # Test find_optimal_k
    results = find_optimal_k(X_test, max_k=8, method='silhouette')
    print(f"Optimal K: {results['optimal_k']}")
    
    # Test kmeans_clustering
    labels, model = kmeans_clustering(X_test, n_clusters=3)
    print(f"Cluster labels shape: {labels.shape}")
    print(f"Number of clusters: {model.n_clusters}")
    
    print("✅ Test passed!")
