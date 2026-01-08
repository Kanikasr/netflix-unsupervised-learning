# 🎬 Netflix Content Promotion & Discovery Decision-Support System

A **platform-level analytics and decision-support system** built using
**exploratory data analysis, unsupervised learning, and decision policy logic**
to support content promotion and discovery strategy for a Netflix-like platform.

---

## 📌 Why This Project Exists

Modern streaming platforms suffer from:
- over-promotion of similar content
- discovery fatigue
- short-term engagement optimization at the cost of long-term satisfaction

This project focuses on the **decision layer above recommendation models**:
how platforms decide **what content to promote**, **how aggressively**, and
**when discovery health is at risk**.

---

##  Dataset Overview

- Netflix Movies & TV Shows catalog
- ~7,800 titles
- Metadata + free-text descriptions
- No user behavior, no labels

---

##  Phase 1 — Exploratory Data Analysis (EDA)

EDA was used not just for visualization, but to **inform modeling decisions**.

### Content Composition
- Movies vs TV Shows distribution reveals a movie-heavy legacy catalog
- Temporal analysis shows increasing dominance of TV content in recent years

![img_2.png](img_2.png)
---

### Duration & Commitment Patterns
- Movies show a wide duration spread (short to very long)
- TV shows cluster around season counts
- Long-duration content is structurally riskier for promotion

![img_3.png](img_3.png)

---

### Genre Structure
- Titles frequently belong to multiple genres
- Genre overlap is high, making single-genre labeling insufficient
- Supports the use of unsupervised learning over rule-based classification

![img_4.png](img_4.png)
---

### Content Staleness
- Significant delay observed between release year and platform addition
- Older content remains present and relevant
- Delay becomes a key risk signal

![img_5.png](img_5.png)
---

### Geographic & Rating Diversity
- Country vs content-type heatmap highlights catalog heterogeneity
- Supports platform-level rather than personalized analysis

![img_6.png](img_6.png)


---

### Text Signal Validation
- Word cloud confirms meaningful semantic signal in descriptions
- Supports NLP-based representation

![img_8.png](img_8.png)

---

##  Phase 2 — Content Representation & Feature Engineering

### Text Processing
- Contraction expansion
- Lowercasing, punctuation & digit removal
- Stopword removal
- Lemmatization (not stemming)

### Vectorization
- TF-IDF with unigrams + bigrams
- Vocabulary capped at 5,000 features

### Dimensionality Reduction
- Truncated SVD (50 components)
- Chosen over PCA due to sparsity
- Preserves semantic structure while reducing noise

---

##  Phase 3 — Latent Content Themes (Clustering)

KMeans clustering (k=4) was selected using:
- Elbow method
- Silhouette score
- Interpretability

![img_9.png](img_9.png)

Clusters represent **latent content themes**, not predefined genres:

| Cluster | Interpretation |
|------|---------------|
| 0 | Recent, mainstream, short-format content |
| 1 | Mixed-era, moderate-duration content |
| 2 | Older, long-form, niche or classic content |
| 3 | TV-heavy, serialized, binge-oriented content |

Cluster redundancy is treated as **signal**, not error.

---

##  Phase 4 — Promotion Failure Risk Modeling

A heuristic **Promotion Failure Risk Score** estimates the likelihood
that a title performs poorly when aggressively promoted.

Inputs:
- Duration (viewer commitment)
- Atypicality within cluster
- Content staleness

This is **preventive analytics**, not prediction.

---

##  Phase 5 — Discovery Diversity Health

Discovery health is audited using:
- Entropy
- Top-1 and Top-2 cluster dominance

Outputs:
- LOW / MEDIUM / HIGH diversity risk

This helps identify content bubbles and discovery fatigue.

---

##  Phase 6 — Hybrid Decision Engine

The Hybrid Engine balances:
- Short-term promotion safety
- Long-term discovery health

It supports **strategy-based scenarios**:
- Conservative
- Balanced
- Exploratory

Exploration is **selective and safety-bounded**.
Redundant or high-risk clusters may be intentionally skipped.

---

##  Streamlit Application

A business-facing dashboard provides:
- Promotion Risk Monitor
- Discovery Diversity Health
- Hybrid Promotion Simulator

Designed for **internal decision-makers**, not end users.

![img_10.png](img_10.png)

---

##  What This Project Does NOT Do

- No user-level personalization
- No watch-time or CTR prediction

---

##  Future Scope

- User-level exposure saturation control
- Feedback loop mitigation
- Boredom-aware exploration policies

---

##  Key Takeaway

This project demonstrates how **EDA + unsupervised learning + policy logic**
can support **real-world content strategy decisions**
without user data or labels.
