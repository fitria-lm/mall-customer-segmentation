"""
Utility functions untuk Mall Customer Segmentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

# ===== FUNGSI LOGGING & PROGRESS =====
def log_message(message, level="INFO"):
    """Print log dengan timestamp dan level"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    colors = {
        "INFO": "\033[94m",    # Blue
        "SUCCESS": "\033[92m", # Green
        "WARNING": "\033[93m", # Yellow
        "ERROR": "\033[91m",   # Red
        "END": "\033[0m"       # Reset
    }
    color = colors.get(level, "\033[94m")
    print(f"{color}[{timestamp}] {level}: {message}{colors['END']}")

def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Print progress bar di terminal
    
    Contoh:
    [███████████████████████████████     ] 85.0% Complete
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total: 
        print()

# ===== FUNGSI VALIDASI DATA =====
def validate_dataframe(df, required_columns=None, min_rows=1):
    """
    Validasi DataFrame untuk memastikan data sesuai ekspektasi
    
    Args:
        df: DataFrame untuk divalidasi
        required_columns: List kolom yang wajib ada
        min_rows: Minimum jumlah baris
    
    Returns:
        bool: True jika valid
    Raises:
        ValueError: Jika tidak valid
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input harus berupa pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame kosong")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame harus memiliki minimal {min_rows} baris, memiliki {len(df)}")
    
    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Kolom yang hilang: {missing}")
    
    log_message(f"DataFrame valid: {df.shape[0]} baris, {df.shape[1]} kolom", "SUCCESS")
    return True

def check_missing_values(df, threshold=0.5):
    """
    Cek missing values dan berikan rekomendasi
    
    Returns:
        dict: Statistik missing values
    """
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    missing_percentage = (missing_cells / total_cells) * 100
    
    result = {
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_percentage": missing_percentage,
        "status": "OK"
    }
    
    if missing_cells > 0:
        log_message(f"Missing values ditemukan: {missing_cells} ({missing_percentage:.2f}%)", "WARNING")
        
        # Tampilkan kolom dengan missing values
        missing_by_column = df.isnull().sum()
        missing_by_column = missing_by_column[missing_by_column > 0]
        
        if not missing_by_column.empty:
            print("\nMissing values per kolom:")
            for col, count in missing_by_column.items():
                percentage = (count / len(df)) * 100
                print(f"  {col}: {count} ({percentage:.1f}%)")
        
        if missing_percentage > threshold:
            result["status"] = "CRITICAL"
            log_message(f"Missing values melebihi threshold {threshold}%", "ERROR")
    
    return result

# ===== FUNGSI IO (SAVE/LOAD) =====
def save_results(df, filename, include_timestamp=True):
    """
    Simpan hasil ke CSV dengan backup otomatis
    
    Args:
        df: DataFrame untuk disimpan
        filename: Nama file output
        include_timestamp: Tambah timestamp ke nama file
    """
    import os
    
    if include_timestamp:
        name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}{ext}"
    
    # Buat folder jika belum ada
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Simpan ke CSV
    df.to_csv(filename, index=False)
    log_message(f"Results saved to: {filename}", "SUCCESS")
    return filename

def save_model_artifacts(model, scaler, filename_prefix="model"):
    """
    Simpan model dan scaler untuk penggunaan ulang
    
    Args:
        model: Model machine learning
        scaler: Scaler yang digunakan
        filename_prefix: Prefix nama file
    """
    import joblib
    import os
    
    os.makedirs("models", exist_ok=True)
    
    # Simpan model
    model_path = f"models/{filename_prefix}_kmeans.pkl"
    joblib.dump(model, model_path)
    log_message(f"Model saved to: {model_path}", "SUCCESS")
    
    # Simpan scaler
    scaler_path = f"models/{filename_prefix}_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    log_message(f"Scaler saved to: {scaler_path}", "SUCCESS")
    
    # Simpan metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "model_type": type(model).__name__,
        "n_clusters": model.n_clusters if hasattr(model, 'n_clusters') else None
    }
    
    metadata_path = f"models/{filename_prefix}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return model_path, scaler_path

# ===== FUNGSI FORMATTING & HELPER =====
def format_currency(amount, currency="$"):
    """Format angka menjadi currency"""
    if amount >= 1000:
        return f"{currency}{amount:,.0f}"
    return f"{currency}{amount:.0f}"

def format_percentage(value, decimals=1):
    """Format angka menjadi percentage"""
    return f"{value:.{decimals}f}%"

def describe_clusters(df, cluster_col='Cluster'):
    """
    Buat summary statistik untuk setiap cluster
    
    Returns:
        DataFrame: Statistik per cluster
    """
    if cluster_col not in df.columns:
        raise ValueError(f"Kolom {cluster_col} tidak ditemukan")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cluster_col in numeric_cols:
        numeric_cols.remove(cluster_col)
    
    cluster_stats = df.groupby(cluster_col)[numeric_cols].agg(['mean', 'std', 'count'])
    
    # Flatten multi-index columns
    cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
    
    # Tambah persentase
    total = len(df)
    cluster_stats['percentage'] = (cluster_stats[[col for col in cluster_stats.columns if 'count' in col]].iloc[:, 0] / total * 100)
    
    log_message(f"Cluster statistics generated for {len(cluster_stats)} clusters", "INFO")
    return cluster_stats

# ===== FUNGSI INISIALISASI =====
def setup_project():
    """Setup awal project: buat folder yang diperlukan"""
    folders = ['data/raw', 'data/processed', 'models', 'reports/figures']
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        log_message(f"Created folder: {folder}", "INFO")
    
    log_message("Project setup completed", "SUCCESS")

# Jika file dijalankan langsung, jalankan setup
if __name__ == "__main__":
    setup_project()
