import os

#Dữ liệu
RFM_FEATURES: list[str] = ["Recency", "Frequency", "Monetary"]
INPUT_FILE: str = os.path.join("input", "customer_rfm_small.csv")
OUTPUT_FILE: str = os.path.join("output", "output_results.csv")

#Tiền xử lý
PCA_COMPONENTS: int = 2
RANDOM_STATE: int = 42

#Phân cụm chung
N_CLUSTERS: int = 4

#DBSCAN
DBSCAN_EPS: float = 0.5
DBSCAN_MIN_SAMPLES: int = 5
DBSCAN_ALGORITHM: str = "ball_tree"

#AutoEncoder
AUTOENCODER_LATENT_DIM: int = 2
AUTOENCODER_HIDDEN_DIM: int = 8
AUTOENCODER_EPOCHS: int = 30
AUTOENCODER_BATCH_SIZE: int = 32
AUTOENCODER_LEARNING_RATE: float = 0.01

#MLP
MLP_HIDDEN_LAYERS: tuple[int, ...] = (64, 32)
MLP_MAX_ITER: int = 300
