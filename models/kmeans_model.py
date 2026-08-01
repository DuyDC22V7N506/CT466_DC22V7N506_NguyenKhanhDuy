import numpy as np
from sklearn.cluster import KMeans

from config import N_CLUSTERS, RANDOM_STATE

def run_kmeans(
    X: np.ndarray,
    n_clusters: int = N_CLUSTERS,
    init: str = "random",
) -> np.ndarray:
    model = KMeans(n_clusters=n_clusters, init=init, random_state=RANDOM_STATE)
    return model.fit_predict(X)

def run_kmeans_pp(
    X: np.ndarray,
    n_clusters: int = N_CLUSTERS,
) -> np.ndarray:
    return run_kmeans(X, n_clusters=n_clusters, init="k-means++")