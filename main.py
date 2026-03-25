from preprocessing import load_and_process
from utils.evaluation import evaluate_model

from models.kmeans_model import run_kmeans
from models.kmeans_pp_model import run_kmeans_pp
from models.dbscan_model import run_dbscan
from models.hierarchical_model import run_hierarchical
from models.gmm_model import run_gmm
from models.autoencoder_kmeans import run_autoencoder_kmeans
from models.mlp_model import run_mlp

# Load data
X, df = load_and_process("data.xlsx")

# Run models------------
kmeans_labels = run_kmeans(X)
evaluate_model(X, kmeans_labels, "KMeans")

kmeans_pp_labels = run_kmeans_pp(X)
evaluate_model(X, kmeans_pp_labels, "KMeans++")

dbscan_labels = run_dbscan(X)
evaluate_model(X, dbscan_labels, "DBSCAN")

hier_labels = run_hierarchical(X)
evaluate_model(X, hier_labels, "Hierarchical")

gmm_labels = run_gmm(X)
evaluate_model(X, gmm_labels, "GMM")

ae_labels = run_autoencoder_kmeans(X)
evaluate_model(X, ae_labels, "AutoEncoder+KMeans")

# MLP dùng pseudo-label từ KMeans
mlp_preds = run_mlp(X, kmeans_labels)

# Save results
output = df.copy()
output['KMeans'] = kmeans_labels
output['KMeans++'] = kmeans_pp_labels
output['DBSCAN'] = dbscan_labels
output['Hierarchical'] = hier_labels
output['GMM'] = gmm_labels
output['AutoEncoder'] = ae_labels
output['MLP'] = mlp_preds

output.to_csv("output_results.csv", index=False)

print("DONE!")