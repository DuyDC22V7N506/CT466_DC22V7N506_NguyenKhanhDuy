"""
preprocessing.py
----------------
Pipeline tiền xử lý dữ liệu RFM (Recency, Frequency, Monetary).

Quy trình:
    1. Đọc CSV → pd.DataFrame
    2. Log-transform để xử lý phân phối lệch (skewed)
    3. Chuẩn hóa (StandardScaler) → np.ndarray 3D
    4. Giảm chiều (PCA 2D) → np.ndarray 2D
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import PCA_COMPONENTS, RANDOM_STATE, RFM_FEATURES


def load_rfm_data(file_path: str) -> pd.DataFrame:
    """Đọc file CSV chứa dữ liệu RFM đã được tính toán sẵn.

    Args:
        file_path: Đường dẫn đến file CSV.

    Returns:
        DataFrame chứa cột Recency, Frequency, Monetary.
    """
    return pd.read_csv(file_path)


def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Áp dụng log1p để giảm độ lệch (skewness) của phân phối RFM.

    Dùng log1p thay vì log để tránh lỗi với giá trị 0.

    Args:
        df: DataFrame chứa các cột RFM_FEATURES.

    Returns:
        DataFrame mới với các cột RFM đã được biến đổi log.
    """
    df_log = df.copy()
    for feature in RFM_FEATURES:
        df_log[feature] = np.log1p(df_log[feature])
    return df_log


def scale_features(df: pd.DataFrame) -> np.ndarray:
    """Chuẩn hóa các đặc trưng RFM về phân phối chuẩn (mean=0, std=1).

    Args:
        df: DataFrame chứa các cột RFM_FEATURES sau khi log-transform.

    Returns:
        Mảng numpy 2D đã chuẩn hóa, shape (n_samples, 3).
    """
    scaler = StandardScaler()
    return scaler.fit_transform(df[RFM_FEATURES])


def reduce_dimensions(X_scaled: np.ndarray) -> np.ndarray:
    """Giảm chiều dữ liệu bằng PCA phục vụ vẽ biểu đồ và phân cụm.

    Args:
        X_scaled: Mảng numpy 2D đã chuẩn hóa, shape (n_samples, n_features).

    Returns:
        Mảng numpy 2D sau PCA, shape (n_samples, PCA_COMPONENTS).
    """
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    return pca.fit_transform(X_scaled)


def load_and_preprocess_rfm(
    file_path: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Pipeline tổng hợp: đọc, biến đổi log, chuẩn hóa, giảm chiều.

    Args:
        file_path: Đường dẫn đến file CSV RFM.

    Returns:
        Tuple gồm:
            - X_pca (np.ndarray): Dữ liệu 2D sau PCA — dùng cho hầu hết models.
            - X_scaled (np.ndarray): Dữ liệu 3D đã chuẩn hóa — dùng cho AutoEncoder.
            - df_rfm (pd.DataFrame): DataFrame gốc — dùng để lưu kết quả.
    """
    df_rfm = load_rfm_data(file_path)
    df_log = apply_log_transform(df_rfm)
    X_scaled = scale_features(df_log)
    X_pca = reduce_dimensions(X_scaled)
    return X_pca, X_scaled, df_rfm