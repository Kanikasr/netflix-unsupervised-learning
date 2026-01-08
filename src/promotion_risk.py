import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# --------------------------------------------------
# A) Duration Risk
# --------------------------------------------------
def compute_duration_risk(df: pd.DataFrame) -> np.ndarray:
    """
    Longer content implies higher commitment and higher promotion risk.
    Movies are penalized more than TV shows.
    """
    duration = df['duration_int'].copy()

    # fill missing with median
    duration.fillna(duration.median(), inplace=True)

    # normalize duration
    duration_norm = (duration - duration.min()) / (duration.max() - duration.min())

    # movie penalty
    is_movie = (df['duration_type'] == 'min').astype(int)

    duration_risk = duration_norm * (1 + 0.3 * is_movie)

    return duration_risk.clip(0, 1)


# --------------------------------------------------
# B) Cluster Atypicality Risk
# --------------------------------------------------
def compute_cluster_distance_risk(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Measures how far a point is from its cluster centroid.
    Farther = more atypical = higher risk.
    """
    distances = np.zeros(len(X))

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        centroid = X[idx].mean(axis=0)
        distances[idx] = np.linalg.norm(X[idx] - centroid, axis=1)

    # normalize distances
    scaler = MinMaxScaler()
    distance_risk = scaler.fit_transform(distances.reshape(-1, 1)).flatten()

    return distance_risk


# --------------------------------------------------
# C) Delay Risk
# --------------------------------------------------
def compute_delay_risk(df: pd.DataFrame) -> np.ndarray:
    """
    Larger gap between release and platform addition increases promotion risk.
    """
    delay = df['delay_years'].copy()
    delay.fillna(delay.median(), inplace=True)

    delay_norm = (delay - delay.min()) / (delay.max() - delay.min())
    return delay_norm.clip(0, 1)


# --------------------------------------------------
# FINAL PROMOTION FAILURE SCORE
# --------------------------------------------------
def compute_promotion_failure_score(
    df: pd.DataFrame,
    X: np.ndarray,
    cluster_col: str = 'km_cluster',
    w_duration: float = 0.4,
    w_cluster: float = 0.4,
    w_delay: float = 0.2
) -> pd.DataFrame:
    """
    Combines duration risk, cluster atypicality risk, and delay risk
    into a single Promotion Failure Score.
    """

    duration_risk = compute_duration_risk(df)
    cluster_risk = compute_cluster_distance_risk(X, df[cluster_col].values)
    delay_risk = compute_delay_risk(df)

    pfs = (
        w_duration * duration_risk +
        w_cluster * cluster_risk +
        w_delay * delay_risk
    )

    df['promotion_failure_score'] = pfs.clip(0, 1)

    return df
