"""
Mall Customer Segmentation - Python Package

Fungsi utama:
- Preprocessing data pelanggan
- Clustering dengan K-Means
- Visualisasi hasil segmentasi
"""

# ===== Ekspos fungsi utama untuk kemudahan import =====
# Contoh: dari src import load_data, kmeans_clustering
from .data_preprocessing import load_data, preprocess_data
from .clustering import kmeans_clustering, find_optimal_k
from .visualization import plot_clusters, plot_elbow_method
from .exploratory_analysis import (
    describe_data, 
    plot_distributions, 
    plot_correlation_heatmap
)

# ===== Metadata package =====
__version__ = "1.0.0"
__author__ = "Nama Anda"
__email__ = "email@anda.com"

# Bisa juga kosong jika tidak mau export spesifik
# File __init__.py kosong juga valid