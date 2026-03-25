from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.cluster import KMeans


def run_autoencoder_kmeans(X, k=4):
    input_dim = X.shape[1]

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(8, activation='relu')(input_layer)
    encoded = Dense(4, activation='relu')(encoded)
    decoded = Dense(8, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)

    autoencoder.compile(optimizer=Adam(), loss='mse')
    autoencoder.fit(X, X, epochs=20, batch_size=32, verbose=0)

    X_encoded = encoder.predict(X)

    kmeans = KMeans(n_clusters=k, random_state=42)
    return kmeans.fit_predict(X_encoded)