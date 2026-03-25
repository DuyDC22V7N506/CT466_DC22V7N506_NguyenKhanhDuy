import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_and_process(path):
    df = pd.read_excel(path)

    # Chỉ lấy cột số
    df = df.select_dtypes(include=[np.number]).dropna()

    # GIẢM SỐ DÒNG (quan trọng)
    if len(df) > 5000:
        df = df.sample(n=5000, random_state=42)

    # Scale dữ liệu
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    # GIẢM CHIỀU (cực quan trọng cho DBSCAN)
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)

    return X, df