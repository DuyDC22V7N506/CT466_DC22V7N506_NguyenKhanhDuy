import numpy as np
from sklearn.mixture import GaussianMixture

from config import N_CLUSTERS, RANDOM_STATE

def run_gmm(
    X: np.ndarray,
    n_clusters: int = N_CLUSTERS,
) -> np.ndarray:
    model = GaussianMixture(n_components=n_clusters, random_state=RANDOM_STATE)
    return model.fit_predict(X)