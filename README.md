# Netflix Content Promotion & Discovery Decision-Support System

This project builds a **platform-level analytics and decision-support system**
for a Netflix-like streaming service using **unsupervised learning**.


---

##  Project Objective

Help **Content Strategy and Product Analytics teams** decide:

- What content is **risky to promote**
- Whether the catalog is becoming **over-concentrated**
- How to balance **promotion safety vs discovery exploration**

---

##  Core Ideas

### 1. Content Understanding (Unsupervised Learning)
- TF-IDF + Truncated SVD for semantic representation
- KMeans clustering to discover **latent content themes**
- No labels, no supervised targets

### 2. Promotion Failure Risk
A heuristic risk score estimating likelihood of poor engagement if content
is aggressively promoted.

Based on:
- Content duration
- Atypicality within cluster
- Content staleness

### 3. Discovery Diversity Health
Audits how balanced content exposure is using:
- Entropy
- Cluster dominance

Flags risks of discovery fatigue and content bubbles.

### 4. Hybrid Decision Engine
Balances:
- Short-term engagement safety
- Long-term discovery health

Supports strategy-based scenarios:
- Conservative
- Balanced
- Exploratory

---

##  Streamlit Application

The project includes a **business-facing Streamlit dashboard** with:

- Promotion Risk Monitor
- Discovery Diversity Health
- Hybrid Content Promotion Simulator
- Strategy-based scenario selection

This tool is designed for **internal stakeholders**, not end users.

---

##  What This Project Is NOT

-  Not a recommender system
-  Not user-level personalization
-  Not predicting watch time or clicks

---

##  Future Scope

- User-level exposure saturation control
- Feedback loop mitigation
- Boredom-aware exploration policies

---

## ️ How to Run

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```