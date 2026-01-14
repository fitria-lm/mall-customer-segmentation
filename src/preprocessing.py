import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def preprocess_data(df, config):
    df_processed = df.copy()
    
    if config['features']['id_column'] in df_processed.columns:
        df_processed = df_processed.drop(config['features']['id_column'], axis=1)
    
    le = LabelEncoder()
    for col in config['features']['categorical']:
        if col in df_processed.columns:
            df_processed[col] = le.fit_transform(df_processed[col])
    
    numerical_cols = config['features']['numerical']
    scaler = StandardScaler()
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    
    joblib.dump(scaler, f"{config['paths']['models']}/scaler.pkl")
    
    return df_processed, scaler