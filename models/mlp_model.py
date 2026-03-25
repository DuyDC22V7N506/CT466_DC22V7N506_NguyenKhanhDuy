from sklearn.neural_network import MLPClassifier


def run_mlp(X, y):
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300)
    model.fit(X, y)
    return model.predict(X)