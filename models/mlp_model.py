import numpy as np
from sklearn.neural_network import MLPClassifier

from config import MLP_HIDDEN_LAYERS, MLP_MAX_ITER, RANDOM_STATE

def run_mlp(
    X: np.ndarray,
    pseudo_labels: np.ndarray,
) -> np.ndarray:
    model = MLPClassifier(
        hidden_layer_sizes=MLP_HIDDEN_LAYERS,
        max_iter=MLP_MAX_ITER,
        random_state=RANDOM_STATE,
    )
    model.fit(X, pseudo_labels)
    return model.predict(X)