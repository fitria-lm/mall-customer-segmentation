# config.py
import os
from pathlib import Path

# Path configurations
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Mall_Customers.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "mall_customers_clustered.csv"

# Clustering parameters
CLUSTERING_CONFIG = {
    'n_clusters': 5,
    'random_state': 42,
    'init': 'k-means++',
    'n_init': 10,
    'max_iter': 300
}

# Feature configurations
FEATURES = {
    'numerical': ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
    'categorical': ['Genre'],
    'target': None,  # Unsupervised learning
    'to_drop': ['CustomerID']
}

# Visualization settings
VISUALIZATION_CONFIG = {
    'figsize': (12, 8),
    'dpi': 100,
    'style': 'seaborn',
    'color_palette': 'Set2'
}

# Model paths
MODEL_PATHS = {
    'scaler': 'models/scaler.pkl',
    'kmeans': 'models/kmeans_model.pkl'
}