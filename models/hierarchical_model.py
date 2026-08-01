import numpy as np
from sklearn.cluster import AgglomerativeClustering

from config import N_CLUSTERS

def run_hierarchical(
    X: np.ndarray,
    n_clusters: int = N_CLUSTERS,
) -> np.ndarray:
    model = AgglomerativeClustering(n_clusters=n_clusters)
    return model.fit_predict(X)