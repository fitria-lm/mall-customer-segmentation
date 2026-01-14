from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def find_optimal_k(data, k_range):
    wcss = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        
        if k > 1:
            score = silhouette_score(data, kmeans.labels_)
            silhouette_scores.append(score)
    
    optimal_k = k_range[np.argmax(silhouette_scores)]
    return optimal_k, wcss, silhouette_scores

def apply_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    return kmeans, labels