"""
utils/evaluation.py
--------------------
Các hàm đánh giá chất lượng phân cụm.

Silhouette Score đo mức độ gắn kết nội cụm và tách biệt liên cụm,
nằm trong khoảng [-1, 1]; giá trị càng cao thì phân cụm càng tốt.
"""

import numpy as np
from sklearn.metrics import silhouette_score


def compute_silhouette_score(
    X: np.ndarray,
    labels: np.ndarray,
) -> float | None:
    """Tính Silhouette Score, tự động loại bỏ điểm nhiễu của DBSCAN.

    Args:
        X: Mảng numpy 2D đặc trưng, shape (n_samples, n_features).
        labels: Mảng nhãn cụm, shape (n_samples,).
            Điểm nhiễu DBSCAN có nhãn -1 sẽ bị loại trước khi tính.

    Returns:
        Silhouette Score dạng float nếu có đủ ≥ 2 cụm hợp lệ,
        hoặc None nếu không thể tính (quá ít cụm).
    """
    labels_arr = np.asarray(labels)
    X_arr = np.asarray(X)

    # Lọc bỏ điểm nhiễu (-1) trước khi tính
    valid_mask = labels_arr != -1
    valid_labels = labels_arr[valid_mask]
    X_valid = X_arr[valid_mask]

    if len(set(valid_labels)) < 2:
        return None

    return silhouette_score(X_valid, valid_labels)


def evaluate_model(
    X: np.ndarray,
    labels: np.ndarray,
    model_name: str,
) -> float | None:
    """Tính và in Silhouette Score cho một model phân cụm.

    Args:
        X: Mảng numpy 2D đặc trưng, shape (n_samples, n_features).
        labels: Mảng nhãn cụm từ model, shape (n_samples,).
        model_name: Tên hiển thị của model trong output.

    Returns:
        Silhouette Score nếu tính được, None nếu không đủ cụm.
    """
    score = compute_silhouette_score(X, labels)

    if score is not None:
        print(f"{model_name} Silhouette Score: {score:.4f}")
    else:
        print(f"{model_name}: Không thể tính Silhouette (quá ít cụm hợp lệ)")

    return score
