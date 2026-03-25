from sklearn.cluster import DBSCAN

def run_dbscan(X):
    model = DBSCAN(eps=0.7, min_samples=5, algorithm='ball_tree')
    return model.fit_predict(X)