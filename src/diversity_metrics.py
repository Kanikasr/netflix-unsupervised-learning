import numpy as np
import pandas as pd
from scipy.stats import entropy


# --------------------------------------------------
# A) Cluster Exposure Distribution
# --------------------------------------------------
def cluster_exposure_distribution(df: pd.DataFrame, cluster_col: str):
    counts = df[cluster_col].value_counts().sort_index()
    probs = counts / counts.sum()
    return probs


# --------------------------------------------------
# B) Diversity (Entropy)
# --------------------------------------------------
def compute_entropy(probs: pd.Series) -> float:
    """
    Shannon entropy of cluster exposure distribution.
    """
    return entropy(probs, base=2)


# --------------------------------------------------
# C) Dominance Ratio
# --------------------------------------------------
def compute_dominance_ratio(probs: pd.Series, top_k: int = 1) -> float:
    """
    Fraction of exposure from top-k clusters.
    """
    return probs.sort_values(ascending=False).head(top_k).sum()


# --------------------------------------------------
# D) Discovery Diversity Risk Assessment
# --------------------------------------------------
def assess_discovery_diversity_risk(
    df: pd.DataFrame,
    cluster_col: str = 'km_cluster'
):
    """
    Evaluates diversity risk using entropy and dominance.
    """

    probs = cluster_exposure_distribution(df, cluster_col)

    ent = compute_entropy(probs)
    dominance_1 = compute_dominance_ratio(probs, top_k=1)
    dominance_2 = compute_dominance_ratio(probs, top_k=2)

    # Heuristic thresholds (explainable)
    if dominance_1 > 0.65:
        risk_flag = "HIGH"
    elif dominance_1 > 0.45:
        risk_flag = "MEDIUM"
    else:
        risk_flag = "LOW"

    return {
        "cluster_distribution": probs,
        "entropy": ent,
        "top1_dominance": dominance_1,
        "top2_dominance": dominance_2,
        "diversity_risk": risk_flag
    }
