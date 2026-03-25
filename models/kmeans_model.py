from sklearn.cluster import KMeans


def run_kmeans(X, k=4):
    model = KMeans(n_clusters=k, random_state=42)
    return model.fit_predict(X)