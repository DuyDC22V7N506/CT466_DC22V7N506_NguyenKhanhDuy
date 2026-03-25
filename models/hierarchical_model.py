from sklearn.cluster import AgglomerativeClustering


def run_hierarchical(X, k=4):
    model = AgglomerativeClustering(n_clusters=k)
    return model.fit_predict(X)