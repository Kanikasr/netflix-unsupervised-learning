import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def build_feature_matrix(df: pd.DataFrame, X_svd: np.ndarray):
    num_cols = ['release_year', 'duration_int', 'num_genres', 'delay_years']

    X_struct = df[num_cols].copy()
    for c in num_cols:
        X_struct[c] = pd.to_numeric(X_struct[c], errors='coerce')
        median_val = X_struct[c].median()
        X_struct[c] = X_struct[c].fillna(median_val)

    scaler = StandardScaler()
    X_struct_scaled = scaler.fit_transform(X_struct)

    X = np.hstack([X_svd, X_struct_scaled])
    return X


def run_kmeans(df: pd.DataFrame, X: np.ndarray, k: int = 4):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    df['km_cluster'] = kmeans.fit_predict(X)

    sil = silhouette_score(X, df['km_cluster'])
    print(f"KMeans Silhouette (k={k}): {sil:.3f}")

    return df, kmeans
