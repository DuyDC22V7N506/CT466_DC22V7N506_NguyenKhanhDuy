"""
models/mlp_model.py
-------------------
Phân loại bằng Multilayer Perceptron (MLP).

Dùng nhãn pseudo-label từ một model phân cụm (thường là KMeans)
làm nhãn giám sát để huấn luyện MLP classifier.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier

from config import MLP_HIDDEN_LAYERS, MLP_MAX_ITER, RANDOM_STATE


def run_mlp(
    X: np.ndarray,
    pseudo_labels: np.ndarray,
) -> np.ndarray:
    """Huấn luyện và dự đoán nhãn cụm bằng MLP Classifier.

    Dùng pseudo-labels từ một model unsupervised (ví dụ KMeans)
    để chuyển bài toán phân cụm thành bài toán phân loại có giám sát.

    Args:
        X: Mảng numpy 2D đặc trưng đầu vào, shape (n_samples, n_features).
        pseudo_labels: Mảng nhãn giả từ model phân cụm, shape (n_samples,).

    Returns:
        Mảng nhãn dự đoán của MLP, shape (n_samples,).
    """
    model = MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN_LAYERS,
        max_iter=MLP_MAX_ITER,
        random_state=RANDOM_STATE,
    )
    model.fit(X, pseudo_labels)
    return model.predict(X)