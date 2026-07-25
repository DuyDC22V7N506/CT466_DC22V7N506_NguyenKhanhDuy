"""
models/hierarchical_model.py
-----------------------------
Phân cụm phân cấp kết tụ (Agglomerative Hierarchical Clustering).

Xây dựng cây phân cấp bằng cách lần lượt gộp các điểm gần nhau nhất.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from config import N_CLUSTERS


def run_hierarchical(
    X: np.ndarray,
    n_clusters: int = N_CLUSTERS,
) -> np.ndarray:
    """Phân cụm phân cấp kết tụ trên dữ liệu đầu vào.

    Args:
        X: Mảng numpy 2D, shape (n_samples, n_features).
        n_clusters: Số cụm cần tạo. Mặc định lấy từ config.

    Returns:
        Mảng nhãn cụm, shape (n_samples,).
    """
    model = AgglomerativeClustering(n_clusters=n_clusters)
    return model.fit_predict(X)