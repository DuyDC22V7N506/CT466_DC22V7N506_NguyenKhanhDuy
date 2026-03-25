from sklearn.mixture import GaussianMixture


def run_gmm(X, k=4):
    model = GaussianMixture(n_components=k, random_state=42)
    return model.fit_predict(X)