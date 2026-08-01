import numpy as np
from sklearn.cluster import DBSCAN

from config import DBSCAN_ALGORITHM, DBSCAN_EPS, DBSCAN_MIN_SAMPLES

def run_dbscan(
    X: np.ndarray,
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
) -> np.ndarray:
    model = DBSCAN(eps=eps, min_samples=min_samples, algorithm=DBSCAN_ALGORITHM)
    return model.fit_predict(X)