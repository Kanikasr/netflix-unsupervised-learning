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

<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/a61860f3-b1af-4f56-a277-37747af48e81" />

---

### Duration & Commitment Patterns
- Movies show a wide duration spread (short to very long)
- TV shows cluster around season counts
- Long-duration content is structurally riskier for promotion

<img width="1014" height="548" alt="image" src="https://github.com/user-attachments/assets/16e917c1-4e10-4bed-a4b4-a856d801f879" />


---

### Genre Structure
- Titles frequently belong to multiple genres
- Genre overlap is high, making single-genre labeling insufficient
- Supports the use of unsupervised learning over rule-based classification

<img width="1151" height="548" alt="image" src="https://github.com/user-attachments/assets/9b0999f9-ba63-4979-a0f8-3ede62545b0c" />

---

### Content Staleness
- Significant delay observed between release year and platform addition
- Older content remains present and relevant
- Delay becomes a key risk signal

<img width="841" height="548" alt="image" src="https://github.com/user-attachments/assets/b21efddd-91ad-4a7c-a28a-9c36af5ea62a" />

---

### Geographic & Rating Diversity
- Country vs content-type heatmap highlights catalog heterogeneity
- Supports platform-level rather than personalized analysis

<img width="1003" height="625" alt="image" src="https://github.com/user-attachments/assets/88847c58-7a98-497b-8042-279444906e91" />



---

### Text Signal Validation
- Word cloud confirms meaningful semantic signal in descriptions
- Supports NLP-based representation

<img width="950" height="601" alt="image" src="https://github.com/user-attachments/assets/1924fa32-750d-499b-a2a5-e5f8c26e0832" />


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

<img width="1011" height="402" alt="image" src="https://github.com/user-attachments/assets/d9e99b75-c150-481b-b3de-c7210e800301" />


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

<img width="1919" height="967" alt="netflix_streamlit_page1" src="https://github.com/user-attachments/assets/c5b36c00-27d2-4a9f-8720-d4729738e891" />



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
