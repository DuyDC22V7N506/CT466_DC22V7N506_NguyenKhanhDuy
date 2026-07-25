"""
models/kmeans_model.py
----------------------
Phân cụm K-Means với hai chiến lược khởi tạo tâm cụm:
    - 'random'    : Khởi tạo ngẫu nhiên (KMeans gốc)
    - 'k-means++' : Khởi tạo thông minh hơn (KMeans++)
"""

import numpy as np
from sklearn.cluster import KMeans

from config import N_CLUSTERS, RANDOM_STATE


def run_kmeans(
    X: np.ndarray,
    n_clusters: int = N_CLUSTERS,
    init: str = "random",
) -> np.ndarray:
    """Phân cụm K-Means trên dữ liệu đầu vào.

    Args:
        X: Mảng numpy 2D, shape (n_samples, n_features).
        n_clusters: Số cụm cần tìm. Mặc định lấy từ config.
        init: Phương pháp khởi tạo — 'random' hoặc 'k-means++'.
            Dùng 'random' để phân biệt rõ với KMeans++.

    Returns:
        Mảng nhãn cụm, shape (n_samples,).
    """
    model = KMeans(n_clusters=n_clusters, init=init, random_state=RANDOM_STATE)
    return model.fit_predict(X)


def run_kmeans_pp(
    X: np.ndarray,
    n_clusters: int = N_CLUSTERS,
) -> np.ndarray:
    """Phân cụm K-Means++ (khởi tạo thông minh, hội tụ nhanh hơn).

    Args:
        X: Mảng numpy 2D, shape (n_samples, n_features).
        n_clusters: Số cụm cần tìm. Mặc định lấy từ config.

    Returns:
        Mảng nhãn cụm, shape (n_samples,).
    """
    return run_kmeans(X, n_clusters=n_clusters, init="k-means++")