# ============================
# MAIN EXECUTION PIPELINE
# ============================

import numpy as np

# --- NLTK one-time setup ---
from src.nltk_setup import setup_nltk

# --- Core pipeline imports ---
from src.data_loader import load_netflix_data
from src.preprocessing import preprocess_netflix_data
from src.text_features import build_text_embeddings
from src.clustering import build_feature_matrix, run_kmeans
from src.promotion_risk import compute_promotion_failure_score
from src.diversity_metrics import assess_discovery_diversity_risk
from src.hybrid_decision_engine import hybrid_content_selection


# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = "data/netflix.csv"
K_CLUSTERS = 4


def main():
    print("Starting Netflix Content Risk Pipeline...\n")

    # 0️⃣ Ensure NLTK resources are available (runs only once)
    setup_nltk()

    # 1️⃣ Load raw data
    df_raw = load_netflix_data(DATA_PATH)
    print(f"Raw dataset loaded: {df_raw.shape}")

    # 2️⃣ Preprocess & feature engineering
    df = preprocess_netflix_data(df_raw)
    print(f"After preprocessing: {df.shape}")

    # 3️⃣ Text embeddings (TF-IDF + TruncatedSVD)
    X_svd, vectorizer, svd = build_text_embeddings(df)
    print(f"Text embedding shape (SVD): {X_svd.shape}")

    # 4️⃣ Build combined feature matrix
    X = build_feature_matrix(df, X_svd)
    print(f"Final feature matrix shape: {X.shape}")

    # 5️⃣ Clustering (KMeans)
    df, kmeans = run_kmeans(df, X, k=K_CLUSTERS)

    print("\nCluster distribution:")
    print(df['km_cluster'].value_counts())

    # 6️⃣ Promotion Failure Score (Upgrade Layer 1)
    df = compute_promotion_failure_score(df, X)

    print("\nPromotion Failure Score summary:")
    print(df['promotion_failure_score'].describe())

    # 8. Quick sanity check (optional but useful)
    print("\nTop 5 highest-risk titles:")
    print(
        df[['title', 'duration_int', 'promotion_failure_score']]
        .sort_values('promotion_failure_score', ascending=False)
        .head(5)
    )

    # 7️⃣ Discovery Diversity Risk (Upgrade Layer 2)
    diversity_report = assess_discovery_diversity_risk(df)

    print("\nDiscovery Diversity Risk Report:")
    print("Cluster exposure distribution:")
    print(diversity_report['cluster_distribution'])
    print(f"Entropy: {diversity_report['entropy']:.3f}")
    print(f"Top-1 Dominance: {diversity_report['top1_dominance']:.2f}")
    print(f"Top-2 Dominance: {diversity_report['top2_dominance']:.2f}")
    print(f"Diversity Risk Level: {diversity_report['diversity_risk']}")

    # 8️⃣ Hybrid Decision Engine (Final Layer)
    selected_content = hybrid_content_selection(
        df=df,
        diversity_report=diversity_report,
        max_items=10
    )

    print("\nFinal Content Selection (Hybrid Engine):")
    print(
        selected_content[
            ['title', 'km_cluster', 'promotion_failure_score']
        ]
    )

    print("\nPipeline completed successfully.")


# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    main()
