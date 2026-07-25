"""
models/gmm_model.py
-------------------
Phân cụm mềm bằng Gaussian Mixture Model (GMM).

Khác với K-Means (hard assignment), GMM cho phép mỗi điểm
thuộc nhiều cụm với xác suất khác nhau (soft assignment).
"""

import numpy as np
from sklearn.mixture import GaussianMixture

from config import N_CLUSTERS, RANDOM_STATE


def run_gmm(
    X: np.ndarray,
    n_clusters: int = N_CLUSTERS,
) -> np.ndarray:
    """Phân cụm Gaussian Mixture Model trên dữ liệu đầu vào.

    Args:
        X: Mảng numpy 2D, shape (n_samples, n_features).
        n_clusters: Số phân phối Gaussian (cụm). Mặc định lấy từ config.

    Returns:
        Mảng nhãn cụm (hard assignment theo xác suất cao nhất),
        shape (n_samples,).
    """
    model = GaussianMixture(n_components=n_clusters, random_state=RANDOM_STATE)
    return model.fit_predict(X)