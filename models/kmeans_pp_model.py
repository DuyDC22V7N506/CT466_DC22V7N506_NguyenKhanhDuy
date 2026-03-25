from sklearn.cluster import KMeans


def run_kmeans_pp(X, k=4):
    model = KMeans(n_clusters=k, init='k-means++', random_state=42)
    return model.fit_predict(X)
