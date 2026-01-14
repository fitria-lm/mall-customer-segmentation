"""
Module untuk preprocessing data pelanggan mall
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Import dari utils
from .utils import log_message, validate_dataframe, check_missing_values

def load_data(filepath, validate=True):
    """
    Load data dari file CSV
    
    Args:
        filepath: Path ke file CSV
        validate: Validasi data setelah load
    
    Returns:
        DataFrame: Data yang diload
    """
    log_message(f"Loading data from: {filepath}", "INFO")
    
    try:
        df = pd.read_csv(filepath)
        log_message(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns", "SUCCESS")
        
        if validate:
            # Validasi kolom yang diharapkan
            expected_columns = ['CustomerID', 'Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
            validate_dataframe(df, expected_columns)
            
            # Cek missing values
            check_missing_values(df)
        
        return df
    
    except FileNotFoundError:
        log_message(f"File not found: {filepath}", "ERROR")
        raise
    except Exception as e:
        log_message(f"Error loading data: {str(e)}", "ERROR")
        raise

def preprocess_data(df, scale_features=True):
    """
    Preprocess data untuk clustering
    
    Args:
        df: DataFrame input
        scale_features: Whether to scale features
    
    Returns:
        tuple: (df_processed, X_scaled, scaler, features_used)
    """
    log_message("Starting data preprocessing", "INFO")
    
    # Buat copy untuk menghindari modifying original
    df_processed = df.copy()
    
    # 1. Encode categorical variable (Gender)
    log_message("Encoding categorical variable: Genre", "INFO")
    df_processed['Genre'] = df_processed['Genre'].map({'Male': 0, 'Female': 1})
    
    # 2. Handle missing values (jika ada)
    missing_info = check_missing_values(df_processed)
    if missing_info['missing_cells'] > 0:
        log_message(f"Handling {missing_info['missing_cells']} missing values", "WARNING")
        # Untuk numeric columns, isi dengan median
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(
            df_processed[numeric_cols].median()
        )
    
    # 3. Select features untuk clustering
    features = ['Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    log_message(f"Selected features for clustering: {features}", "INFO")
    
    X = df_processed[features].copy()
    
    # 4. Scale features jika diperlukan
    scaler = None
    X_scaled = X.values
    
    if scale_features:
        log_message("Scaling features using StandardScaler", "INFO")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        log_message("Features scaled successfully", "SUCCESS")
    
    log_message(f"Preprocessing complete. Final shape: {X_scaled.shape}", "SUCCESS")
    
    return df_processed, X_scaled, scaler, features

def prepare_data_for_clustering(df, features_to_scale=None):
    """
    Prepare data khusus untuk clustering (versi alternatif)
    
    Args:
        df: DataFrame input
        features_to_scale: List features untuk di-scale (None = semua numeric)
    
    Returns:
        tuple: (X_scaled, scaler, feature_names)
    """
    log_message("Preparing data for clustering", "INFO")
    
    df_processed = df.copy()
    
    # Encode gender jika belum
    if 'Genre' in df_processed.columns and df_processed['Genre'].dtype == 'object':
        df_processed['Genre'] = df_processed['Genre'].map({'Male': 0, 'Female': 1})
    
    # Tentukan features untuk scaling
    if features_to_scale is None:
        # Default: semua kolom numeric kecuali CustomerID
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if 'CustomerID' in numeric_cols:
            numeric_cols.remove('CustomerID')
        features_to_scale = numeric_cols
    
    # Extract features
    X = df_processed[features_to_scale].values
    feature_names = features_to_scale
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    log_message(f"Data prepared for clustering: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features", "SUCCESS")
    
    return X_scaled, scaler, feature_names

# Fungsi untuk test
if __name__ == "__main__":
    # Test fungsi load_data
    test_df = pd.DataFrame({
        'CustomerID': [1, 2, 3],
        'Genre': ['Male', 'Female', 'Male'],
        'Age': [25, 30, 35],
        'Annual Income (k$)': [50, 60, 70],
        'Spending Score (1-100)': [30, 40, 50]
    })
    
    print("Testing preprocessing functions...")
    df_proc, X_scaled, scaler, features = preprocess_data(test_df)
    print(f"Processed DataFrame shape: {df_proc.shape}")
    print(f"Scaled data shape: {X_scaled.shape}")
    print(f"Features used: {features}")
    print("✅ Test passed!")
    