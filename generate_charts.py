
import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from config import RFM_FEATURES, INPUT_FILE, PCA_COMPONENTS, RANDOM_STATE, AUTOENCODER_LATENT_DIM

OUT_DIR = os.path.join(ROOT, "output", "charts")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.dpi":       180,      
    "savefig.dpi":      180,
    "savefig.bbox":     "tight",
    "savefig.facecolor": "white",
})

CLUSTER_COLORS = {
    0: "#4C72B0",   
    1: "#DD8452",  
    2: "#55A868",   
    3: "#C44E52",   
    4: "#8172B2",  
   -1: "#AAAAAA",   
}

CLUSTER_LABELS = {
    0: "Cụm 0",
    1: "Cụm 1",
    2: "Cụm 2",
    3: "Cụm 3",
    4: "Cụm 4",
   -1: "Nhiễu (Noise)",
}

df_results = pd.read_csv(os.path.join(ROOT, "output", "output_results.csv"))
df_rfm     = pd.read_csv(os.path.join(ROOT, INPUT_FILE))

df_log = df_rfm.copy()
for col in RFM_FEATURES:
    df_log[col] = np.log1p(df_log[col])

scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(df_log[RFM_FEATURES])

pca_main  = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
X_pca     = pca_main.fit_transform(X_scaled)          # Trục cho 5 model ML
ev        = pca_main.explained_variance_ratio_
xlabel_pca = f"PC1 ({ev[0]*100:.1f}% phương sai)"
ylabel_pca = f"PC2 ({ev[1]*100:.1f}% phương sai)"

pca_ae   = PCA(n_components=AUTOENCODER_LATENT_DIM, random_state=RANDOM_STATE)
X_latent = pca_ae.fit_transform(X_scaled)
ev_ae    = pca_ae.explained_variance_ratio_
xlabel_ae = f"Latent-1 ({ev_ae[0]*100:.1f}%)"
ylabel_ae = f"Latent-2 ({ev_ae[1]*100:.1f}%)"

def _make_legend(unique_labels: np.ndarray) -> list:
    ordered = sorted([l for l in unique_labels if l != -1])
    if -1 in unique_labels:
        ordered.append(-1)
    return [
        mpatches.Patch(color=CLUSTER_COLORS.get(l, "#999999"),
                       label=CLUSTER_LABELS.get(l, f"Cụm {l}"))
        for l in ordered
    ]

def plot_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    subtitle: str = "",
) -> None:
    unique_labels = np.unique(labels)
    fig, ax = plt.subplots(figsize=(7, 5))

    for lbl in sorted(unique_labels):
        mask  = labels == lbl
        color = CLUSTER_COLORS.get(lbl, "#999999")
        size  = 55 if lbl != -1 else 35
        alpha = 0.85 if lbl != -1 else 0.50
        marker = "o" if lbl != -1 else "x"
        ax.scatter(
            X[mask, 0], X[mask, 1],
            c=color, s=size, alpha=alpha,
            marker=marker, edgecolors="white" if lbl != -1 else "none",
            linewidths=0.4,
        )

    for lbl in unique_labels:
        if lbl == -1:
            continue
        mask = labels == lbl
        cx, cy = X[mask, 0].mean(), X[mask, 1].mean()
        ax.scatter(cx, cy, c="white", s=130, marker="*",
                   edgecolors=CLUSTER_COLORS.get(lbl, "#333"), linewidths=1.2,
                   zorder=5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    if subtitle:
        ax.set_title(title + "\n" + subtitle, pad=10)

    legend_handles = _make_legend(unique_labels)
    ax.legend(handles=legend_handles, loc="best", fontsize=9,
              framealpha=0.85, edgecolor="#cccccc")

    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    out_path = os.path.join(OUT_DIR, filename)
    fig.savefig(out_path)
    plt.close(fig)

def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float | None:
    labels_arr = np.asarray(labels)
    valid      = labels_arr != -1
    if len(set(labels_arr[valid])) < 2:
        return None
    return silhouette_score(X[valid], labels_arr[valid])

models = [
    # (cột trong CSV,   X dùng để vẽ,  xlabel,       ylabel,       tên file,                tiêu đề)
    ("KMeans",      X_pca,    xlabel_pca, ylabel_pca, "hinh_4_1_kmeans.png",
     "Hình 1 – Kết quả phân cụm K-Means"),

    ("KMeans++",    X_pca,    xlabel_pca, ylabel_pca, "hinh_4_2_kmeans_pp.png",
     "Hình 2 – Kết quả phân cụm K-Means++"),

    ("GMM",         X_pca,    xlabel_pca, ylabel_pca, "hinh_4_3_gmm.png",
     "Hình 3 – Kết quả phân cụm GMM"),

    ("Hierarchical",X_pca,    xlabel_pca, ylabel_pca, "hinh_4_4_hierarchical.png",
     "Hình 4 – Kết quả phân cụm Hierarchical (Agglomerative)"),

    ("DBSCAN",      X_pca,    xlabel_pca, ylabel_pca, "hinh_4_5_dbscan.png",
     "Hình 5 – Kết quả phân cụm DBSCAN"),

    ("AutoEncoder", X_latent, xlabel_ae,  ylabel_ae,  "hinh_4_6_autoencoder.png",
     "Hình 6 – Kết quả phân cụm AutoEncoder (PCA + K-Means)"),

    ("MLP",         X_pca,    xlabel_pca, ylabel_pca, "hinh_4_7_mlp.png",
     "Hình 7 – Kết quả phân cụm MLP (Học có giám sát – nhãn giả)"),
]

silhouette_scores: dict[str, float | None] = {}

for col, X_plot, xl, yl, fname, title in models:
    labels = df_results[col].to_numpy()
    score  = compute_silhouette(X_plot, labels)
    silhouette_scores[col] = score

    subtitle = f"Silhouette Score: {score:.4f}" if score is not None else "Silhouette Score: N/A"
    plot_scatter(X_plot, labels, title, xl, yl, fname, subtitle=subtitle)

model_names  = list(silhouette_scores.keys())
score_values = [v if v is not None else 0.0 for v in silhouette_scores.values()]
valid_mask   = [v is not None for v in silhouette_scores.values()]


bar_colors = []
best_idx   = int(np.argmax(score_values))
for i, is_valid in enumerate(valid_mask):
    if not is_valid:
        bar_colors.append("#CCCCCC")
    elif i == best_idx:
        bar_colors.append("#E8A838")   # Vàng — tốt nhất
    else:
        bar_colors.append("#4C72B0")   # Xanh dương — còn lại

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(model_names, score_values, color=bar_colors,
              edgecolor="white", linewidth=0.8, width=0.55)

# Chú thích giá trị trên mỗi cột
for bar, is_valid, val in zip(bars, valid_mask, score_values):
    if is_valid:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )
    else:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.01, "N/A",
            ha="center", va="bottom", fontsize=9, color="#888888",
        )

# Đường tham chiếu phân loại chất lượng
ax.axhline(0.5,  color="#E05C5C", linestyle="--", linewidth=0.9, alpha=0.7, label="Tốt (≥ 0.50)")
ax.axhline(0.25, color="#F0A033", linestyle=":",  linewidth=0.9, alpha=0.7, label="Trung bình (≥ 0.25)")

ax.set_ylim(0, max(score_values) * 1.20 + 0.05)
ax.set_ylabel("Silhouette Score")
ax.set_title("Hình 8 – So sánh Silhouette Score giữa các thuật toán phân cụm",
             pad=12)
ax.set_xlabel("Thuật toán")
ax.legend(fontsize=9, framealpha=0.85, edgecolor="#cccccc")
ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.6)
ax.spines[["top", "right"]].set_visible(False)

# Nền nhạt xen kẽ
for i, bar in enumerate(bars):
    if i % 2 == 0:
        ax.axvspan(bar.get_x() - 0.3, bar.get_x() + bar.get_width() + 0.3,
                   alpha=0.04, color="gray")

out_path = os.path.join(OUT_DIR, "hinh_4_8_silhouette_comparison.png")
fig.savefig(out_path)
plt.close(fig)

print()
