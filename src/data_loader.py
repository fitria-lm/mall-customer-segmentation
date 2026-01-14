import pandas as pd
import numpy as np
import yaml

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_data():
    config = load_config()
    df = pd.read_csv(config['data']['raw_path'])
    return df

def data_summary(df):
    print("Dataset Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDuplicate Rows:", df.duplicated().sum())
    return df