import numpy as np
from sklearn.metrics import silhouette_score

def compute_silhouette_score(
    X: np.ndarray,
    labels: np.ndarray,
) -> float | None:
    labels_arr = np.asarray(labels)
    X_arr = np.asarray(X)

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
    score = compute_silhouette_score(X, labels)

    if score is not None:
        print(f"{model_name} Silhouette Score: {score:.4f}")
    else:
        print(f"{model_name}: Không thể tính Silhouette (quá ít cụm hợp lệ)")

    return score
