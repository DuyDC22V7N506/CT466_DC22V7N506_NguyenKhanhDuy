from sklearn.metrics import silhouette_score


def evaluate_model(X, labels, name):
    if len(set(labels)) > 1:
        score = silhouette_score(X, labels)
        print(f"{name} Silhouette: {score:.4f}")
    else:
        print(f"{name}: Cannot compute silhouette")
