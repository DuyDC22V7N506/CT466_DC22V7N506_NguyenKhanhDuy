"""
models/dbscan_model.py
----------------------
Phân cụm mật độ DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

Ưu điểm: Phát hiện được cụm có hình dạng bất kỳ và tự động xác định điểm nhiễu (-1).
"""

import numpy as np
from sklearn.cluster import DBSCAN

from config import DBSCAN_ALGORITHM, DBSCAN_EPS, DBSCAN_MIN_SAMPLES


def run_dbscan(
    X: np.ndarray,
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
) -> np.ndarray:
    """Phân cụm DBSCAN trên dữ liệu đầu vào.

    Args:
        X: Mảng numpy 2D, shape (n_samples, n_features).
        eps: Bán kính vùng lân cận của mỗi điểm.
            Giá trị 0.5 phù hợp với phân phối mật độ dữ liệu RFM đã scale.
        min_samples: Số điểm tối thiểu trong vùng eps để tạo thành một cụm.

    Returns:
        Mảng nhãn cụm, shape (n_samples,).
        Điểm nhiễu được gán nhãn -1.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples, algorithm=DBSCAN_ALGORITHM)
    return model.fit_predict(X)