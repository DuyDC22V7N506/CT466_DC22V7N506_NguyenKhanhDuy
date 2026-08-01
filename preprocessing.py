import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import PCA_COMPONENTS, RANDOM_STATE, RFM_FEATURES

def load_rfm_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    
    df_log = df.copy()
    for feature in RFM_FEATURES:
        df_log[feature] = np.log1p(df_log[feature])
    return df_log

def scale_features(df: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(df[RFM_FEATURES])

def reduce_dimensions(X_scaled: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    return pca.fit_transform(X_scaled)

def load_and_preprocess_rfm(
    file_path: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df_rfm = load_rfm_data(file_path)
    df_log = apply_log_transform(df_rfm)
    X_scaled = scale_features(df_log)
    X_pca = reduce_dimensions(X_scaled)
    return X_pca, X_scaled, df_rfm