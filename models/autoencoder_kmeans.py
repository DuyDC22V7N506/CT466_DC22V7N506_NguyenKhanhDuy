import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from config import AUTOENCODER_LATENT_DIM, N_CLUSTERS, RANDOM_STATE

def build_latent_encoder(
    latent_dim: int = AUTOENCODER_LATENT_DIM,
) -> PCA:
    return PCA(n_components=latent_dim, random_state=RANDOM_STATE)


def run_autoencoder_kmeans(
    X: np.ndarray,
    n_clusters: int = N_CLUSTERS,
) -> np.ndarray:
    encoder = build_latent_encoder()
    X_latent = encoder.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    return kmeans.fit_predict(X_latent)