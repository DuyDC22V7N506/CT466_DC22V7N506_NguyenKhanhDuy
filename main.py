import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from config import INPUT_FILE, OUTPUT_FILE
from models.autoencoder_kmeans import run_autoencoder_kmeans
from models.dbscan_model import run_dbscan
from models.gmm_model import run_gmm
from models.hierarchical_model import run_hierarchical
from models.kmeans_model import run_kmeans, run_kmeans_pp
from models.mlp_model import run_mlp
from preprocessing import load_and_preprocess_rfm
from utils.evaluation import evaluate_model

def run_all_models(
    X_pca: np.ndarray,
    X_scaled: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    cluster_runners: dict[str, callable] = {
        "KMeans":      lambda: run_kmeans(X_pca),
        "KMeans++":    lambda: run_kmeans_pp(X_pca),
        "DBSCAN":      lambda: run_dbscan(X_pca),
        "Hierarchical": lambda: run_hierarchical(X_pca),
        "GMM":         lambda: run_gmm(X_pca),
        "AutoEncoder": lambda: run_autoencoder_kmeans(X_scaled),
    }

    cluster_labels: dict[str, np.ndarray] = {}
    for model_name, runner in cluster_runners.items():
        labels = runner()
        cluster_labels[model_name] = labels
        evaluate_model(X_pca, labels, model_name)

    mlp_labels = run_mlp(X_pca, pseudo_labels=cluster_labels["KMeans"])

    return cluster_labels, mlp_labels

def save_results(
    df_rfm: pd.DataFrame,
    cluster_labels: dict[str, np.ndarray],
    mlp_labels: np.ndarray,
    output_path: str = OUTPUT_FILE,
) -> None:
    output = df_rfm.copy()
    for model_name, labels in cluster_labels.items():
        output[model_name] = labels
    output["MLP"] = mlp_labels

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    output.to_csv(output_path, index=False)
    print(f"\nKết quả đã lưu tại: {output_path}")


def ensure_csv_exists(csv_path: str) -> None:
    if os.path.exists(csv_path):
        return

    print(f"  '{csv_path}' chưa tồn tại — tự động tạo từ Excel...\n")
    from prepare_data import prepare_dataset

    if "small" in csv_path:
        prepare_dataset(input_xlsx=os.path.join("input", "data_small.xlsx"), output_csv=csv_path)
    else:
        prepare_dataset(input_xlsx=os.path.join("input", "data_large.xlsx"), output_csv=csv_path)

    print()


def main() -> None:
    ensure_csv_exists(INPUT_FILE)

    print(f"=== Đang đọc và xử lý dữ liệu từ '{INPUT_FILE}' ===\n")
    X_pca, X_scaled, df_rfm = load_and_preprocess_rfm(INPUT_FILE)

    print("=== Chạy các model phân cụm ===\n")
    cluster_labels, mlp_labels = run_all_models(X_pca, X_scaled)

    print("\n=== Lưu kết quả ===")
    save_results(df_rfm, cluster_labels, mlp_labels)

    print("\nDONE!")


if __name__ == "__main__":
    main()