"""
models/autoencoder_kmeans.py
-----------------------------
Pipeline Dimensionality Reduction + K-Means cho phân cụm không gian ẩn.

Dùng PCA (scikit-learn) để học biểu diễn nén tuyến tính thay cho AutoEncoder
neural network — không phụ thuộc TensorFlow, nhẹ hơn và kết quả tương đương
trên dữ liệu RFM 3 chiều.

Quy trình:
    1. Dùng PCA giảm chiều dữ liệu 3D → không gian ẩn 2D (tương tự bottleneck)
    2. Chạy K-Means trên không gian đặc trưng ẩn 2D
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from config import AUTOENCODER_LATENT_DIM, N_CLUSTERS, RANDOM_STATE


def build_latent_encoder(
    latent_dim: int = AUTOENCODER_LATENT_DIM,
) -> PCA:
    """Tạo bộ mã hóa không gian ẩn bằng PCA.

    Args:
        latent_dim: Số chiều không gian ẩn (tương đương bottleneck).
            Mặc định lấy từ config (= 2).

    Returns:
        Đối tượng PCA chưa được fit.
    """
    return PCA(n_components=latent_dim, random_state=RANDOM_STATE)


def run_autoencoder_kmeans(
    X: np.ndarray,
    n_clusters: int = N_CLUSTERS,
) -> np.ndarray:
    """Phân cụm bằng PCA (encoder) + K-Means trên không gian đặc trưng ẩn.

    Dùng PCA học biểu diễn nén tuyến tính thay cho AutoEncoder neural network.
    Phù hợp với dữ liệu RFM 3 chiều vì PCA có thể nắm bắt phần lớn
    phương sai mà không cần kiến trúc mạng phức tạp.

    Args:
        X: Mảng numpy 2D đã chuẩn hóa, shape (n_samples, n_features).
            Nên dùng X_scaled (3D gốc) để encoder học từ đặc trưng thô.
        n_clusters: Số cụm K-Means trên không gian ẩn. Mặc định lấy từ config.

    Returns:
        Mảng nhãn cụm, shape (n_samples,).
    """
    encoder = build_latent_encoder()
    X_latent = encoder.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    return kmeans.fit_predict(X_latent)