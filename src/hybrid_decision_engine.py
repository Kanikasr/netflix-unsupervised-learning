import pandas as pd
import numpy as np


# --------------------------------------------------
# HYBRID DECISION ENGINE (FINAL)
# --------------------------------------------------
def hybrid_content_selection(
    df: pd.DataFrame,
    diversity_report: dict,
    cluster_col: str = 'km_cluster',
    risk_col: str = 'promotion_failure_score',
    max_items: int = 10,
    risk_threshold: float = 0.6,
    exploration_bias: str = "MEDIUM"
):
    """
    Selects content using:
    - Promotion Failure Risk (short-term)
    - Discovery Diversity Health (long-term)
    - Strategy-driven exploration bias
    """

    # 1️⃣ Filter high-risk content
    safe_df = df[df[risk_col] <= risk_threshold].copy()

    # Edge case: if too few items remain, relax constraint
    if len(safe_df) < max_items:
        safe_df = df.copy()

    # 2️⃣ Determine exploration ratio (STRATEGY-DRIVEN)
    if exploration_bias == "HIGH":
        exploration_ratio = 0.40
    elif exploration_bias == "LOW":
        exploration_ratio = 0.10
    else:
        # MEDIUM → adapt to diversity health
        if diversity_report['diversity_risk'] == "HIGH":
            exploration_ratio = 0.35
        elif diversity_report['diversity_risk'] == "MEDIUM":
            exploration_ratio = 0.25
        else:
            exploration_ratio = 0.10

    n_explore = int(max_items * exploration_ratio)
    n_exploit = max_items - n_explore

    # 3️⃣ Identify dominant & under-exposed clusters
    cluster_probs = diversity_report['cluster_distribution']
    dominant_cluster = cluster_probs.idxmax()
    under_exposed_clusters = cluster_probs.sort_values().index.tolist()

    # 4️⃣ Exploitation: safe content from dominant cluster
    exploit_candidates = (
        safe_df[safe_df[cluster_col] == dominant_cluster]
        .sort_values(risk_col)
        .head(n_exploit)
    )

    # 5️⃣ Exploration: safe content from under-exposed clusters
    explore_candidates = (
        safe_df[safe_df[cluster_col].isin(under_exposed_clusters)]
        .sort_values(risk_col)
        .head(n_explore)
    )

    # 6️⃣ Combine & finalize
    final_selection = pd.concat([exploit_candidates, explore_candidates])
    final_selection = final_selection.drop_duplicates().head(max_items)

    return final_selection
