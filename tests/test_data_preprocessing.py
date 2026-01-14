# tests/test_data_preprocessing.py
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Tambahkan path ke sys.path untuk import module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import dari src.data_preprocessing (sesuai nama file)
from src.data_preprocessing import (
    load_data,
    clean_data,
    encode_categorical,
    scale_features
)

def test_load_data():
    """Test fungsi load_data dari data_preprocessing.py"""
    # Buat file test dummy
    test_df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c']
    })
    
    # Save ke temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_df.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        # Test load
        loaded_df = load_data(temp_path)
        assert isinstance(loaded_df, pd.DataFrame)
        assert loaded_df.shape == (3, 2)
    finally:
        os.unlink(temp_path)

def test_encode_categorical():
    """Test encoding gender dari data_preprocessing.py"""
    from src.data_preprocessing import preprocess_data
    
    df = pd.DataFrame({
        'Genre': ['Male', 'Female', 'Male'],
        'Age': [25, 30, 35],
        'Annual Income (k$)': [50, 60, 70],
        'Spending Score (1-100)': [30, 40, 50]
    })
    
    result = preprocess_data(df)
    df_processed = result[0]  # Return pertama adalah DataFrame
    
    # Test encoding
    assert df_processed['Genre'].dtype in [np.int64, np.int32]
    assert set(df_processed['Genre'].unique()) == {0, 1}