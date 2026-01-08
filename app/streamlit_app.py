# ============================================================
# STREAMLIT APP: Netflix Content Promotion & Discovery Control
# ============================================================

import sys
import os

# --- Fix import path for Streamlit ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import streamlit as st
import pandas as pd

# --- Core pipeline imports ---
from src.data_loader import load_netflix_data
from src.preprocessing import preprocess_netflix_data
from src.text_features import build_text_embeddings
from src.clustering import build_feature_matrix, run_kmeans
from src.promotion_risk import compute_promotion_failure_score
from src.diversity_metrics import assess_discovery_diversity_risk
from src.hybrid_decision_engine import hybrid_content_selection
from src.nltk_setup import setup_nltk


# ---------------------------
# CONFIG
# ---------------------------
DATA_PATH = "data/netflix.csv"
K_CLUSTERS = 4


# ---------------------------
# STRATEGY PRESETS
# ---------------------------
STRATEGIES = {
    "Conservative": {
        "description": "Minimize user frustration. Avoid risky content. Minimal exploration.",
        "risk_threshold": 0.35,
        "exploration_bias": "LOW"
    },
    "Balanced": {
        "description": "Balance engagement and discovery. Moderate exploration.",
        "risk_threshold": 0.60,
        "exploration_bias": "MEDIUM"
    },
    "Exploratory": {
        "description": "Encourage discovery. Allow higher risk. Strong exploration.",
        "risk_threshold": 0.75,
        "exploration_bias": "HIGH"
    }
}


# ---------------------------
# CLUSTER PROFILES
# ---------------------------
CLUSTER_PROFILES = {
    0: "Recent, mainstream, short-format content (safe & familiar)",
    1: "Mixed-era content with moderate duration and genre variety",
    2: "Older, long-form, niche or classic content (high commitment)",
    3: "TV-heavy, serialized, binge-oriented content"
}


# ---------------------------
# PIPELINE LOADER (CACHED)
# ---------------------------
@st.cache_data(show_spinner=True)
def load_pipeline():
    setup_nltk()

    df_raw = load_netflix_data(DATA_PATH)
    df = preprocess_netflix_data(df_raw)

    X_svd, _, _ = build_text_embeddings(df)
    X = build_feature_matrix(df, X_svd)

    df, _ = run_kmeans(df, X, k=K_CLUSTERS)
    df = compute_promotion_failure_score(df, X)

    diversity_report = assess_discovery_diversity_risk(df)

    return df, X, diversity_report


# ============================================================
# APP UI
# ============================================================

st.set_page_config(
    page_title="Netflix Content Promotion & Discovery Control Panel",
    layout="wide"
)

st.title("🎬 Netflix Content Promotion & Discovery Control Panel")
st.caption(
    "A decision-support system for balancing promotion safety "
    "and discovery diversity in a streaming content catalog."
)

# Load data
df, X, diversity_report = load_pipeline()

# Tabs
tab1, tab2, tab3 = st.tabs([
    "Promotion Risk Monitor",
    "Discovery Diversity Health",
    "Hybrid Decision Simulator"
])


# ============================================================
# TAB 1 — PROMOTION RISK MONITOR
# ============================================================
with tab1:
    st.subheader("Promotion Failure Risk Overview")

    st.markdown(
        """
        This view highlights **content that is structurally risky to promote**.
        Risk is estimated using:
        - content length
        - how atypical it is within its theme
        - how old or delayed the content is
        """
    )

    risk_threshold = st.slider(
        "Promotion Risk Threshold (risk tolerance)",
        min_value=0.10,
        max_value=0.80,
        value=0.60,
        step=0.05
    )

    risky = (
        df[df['promotion_failure_score'] >= risk_threshold]
        .sort_values('promotion_failure_score', ascending=False)
    )

    st.metric(
        "High-Risk Titles",
        len(risky),
        help="Titles likely to cause early disengagement if promoted aggressively."
    )

    st.dataframe(
        risky[['title', 'duration_int', 'promotion_failure_score', 'km_cluster']]
        .head(20)
        .reset_index(drop=True),
        use_container_width=True
    )


# ============================================================
# TAB 2 — DISCOVERY DIVERSITY HEALTH
# ============================================================
with tab2:
    st.subheader("Discovery Diversity Health")

    st.markdown(
        """
        This view audits **how balanced content exposure is** across discovered
        content themes. Over-concentration can reduce long-term discovery
        and perceived catalog variety.
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Entropy (Diversity)", f"{diversity_report['entropy']:.2f}")
    col2.metric("Top-1 Cluster Share", f"{diversity_report['top1_dominance']:.0%}")
    col3.metric("Diversity Risk", diversity_report['diversity_risk'])

    st.subheader("Cluster Exposure Distribution")
    st.bar_chart(diversity_report['cluster_distribution'])


# ============================================================
# TAB 3 — HYBRID DECISION SIMULATOR
# ============================================================
with tab3:
    st.subheader("Hybrid Content Promotion Simulator")

    st.markdown(
        """
        This simulator shows how **different business strategies**
        lead to **different promotion decisions**, even when the
        underlying content catalog stays the same.
        """
    )

    # Strategy selector
    strategy_name = st.radio(
        "Select Promotion Strategy",
        list(STRATEGIES.keys()),
        horizontal=True
    )

    strategy = STRATEGIES[strategy_name]

    st.info(f"**Strategy rationale:** {strategy['description']}")

    col1, col2 = st.columns(2)

    with col1:
        max_items = st.slider(
            "Number of Titles to Promote",
            5, 20, 10
        )

    with col2:
        st.metric(
            "Max Allowed Promotion Risk",
            strategy['risk_threshold']
        )

    # Hybrid decision engine
    selection = hybrid_content_selection(
        df=df,
        diversity_report=diversity_report,
        max_items=max_items,
        risk_threshold=strategy['risk_threshold'],
        exploration_bias=strategy['exploration_bias']
    )

    st.subheader("Recommended Promotion Set")

    st.dataframe(
        selection[['title', 'km_cluster', 'promotion_failure_score']]
        .reset_index(drop=True),
        use_container_width=True
    )

    st.caption(
        "Selection balances promotion safety with discovery exploration "
        "based on the chosen strategy."
    )
# Cluster explanations
with st.expander("ℹ️ How to interpret content clusters"):
    st.markdown(
        """
        Clusters represent **automatically discovered content themes**
        based on content descriptions, genres, duration, and release patterns.
        They are not predefined genres and may span multiple categories.
        """
    )

    for cid, desc in CLUSTER_PROFILES.items():
        st.markdown(f"**Cluster {cid}:** {desc}")

