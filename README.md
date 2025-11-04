# Netflix Shows & Movies: Unsupervised Learning Analysis

<img width="1200" height="675" alt="image" src="https://github.com/user-attachments/assets/8e8484db-8fa0-4cfd-862e-f8f4c9ec30fb" />


## Project Overview
This project explores Netflix’s global catalogue of shows and movies using **unsupervised learning techniques**.  
The goal is to uncover **patterns in content type, release trends, and genre-based themes** — and to understand how Netflix’s content focus has evolved over time.

Dataset sourced from **Flixable (2019)** provides detailed metadata for every title available on Netflix.

---

## Problem Statement
Netflix has shifted its focus in recent years from movies toward TV shows.  
This analysis investigates:
- What type of content dominates across countries.
- How the content trends have evolved over time.
- How similar titles can be grouped using text (genre/description) and numeric metadata.

---

## Dataset Overview
**Columns:**
`['show_id', 'type', 'title', 'director', 'cast', 'country', 'date_added', 'release_year', 'rating', 'duration', 'listed_in', 'description']`

**Key Attributes:**
- **Type:** TV Show / Movie  
- **Title:** Name of the content  
- **Country:** Country of release  
- **Date Added:** Date added to Netflix  
- **Release Year:** Original release year  
- **Duration:** Minutes or number of seasons  
- **Listed In:** Genres  
- **Description:** Short text summary  

---

##  Methodology

### 1️. Data Cleaning
- Handled missing values and duplicates.  
- Standardized duration units (converted all to minutes).  
- Extracted `year`, `month`, and `delay_years` (time gap between release and Netflix addition).  
- Derived `num_genres` from genre column.

### 2️. Text Preprocessing
- Expanded contractions and removed punctuation, digits, and stopwords.  
- Lemmatized text using **WordNetLemmatizer**.  
- Applied **TF-IDF Vectorization** to represent text numerically.

### 3️. Feature Engineering
- Combined TF-IDF text features with numeric features (`release_year`, `duration_int`, `num_genres`, `delay_years`).  
- Scaled all features using **StandardScaler**.

### 4️. Clustering Models
| Model | Technique | Evaluation Metric | Key Observation |
|--------|------------|------------------|----------------|
| **Model 1** | K-Means | Elbow + Silhouette | 4 clusters, Silhouette ≈ 0.45 |
| **Model 2** | Hierarchical (Ward Linkage) | Silhouette | 4 clusters, similar structure to K-Means |
| **Model 3** | DBSCAN | Silhouette + Noise Detection | Many noise points (≈75%), few dense clusters |

---

## Cluster Insights

| Cluster | Description | Example Theme |
|----------|--------------|----------------|
| **Classic Adapted Films** | Older, multi-genre films (2000s) | Historical dramas, adaptations |
| **Modern Feature-Length Dramas** | Recent, emotional, or social films | Contemporary storytelling |
| **Short Web Shows** | Compact, global, recent shows | Web-based short series |
| **Single-Genre Documentaries** | Focused thrillers or factual content | True crime, docs |

**Key takeaway:** Netflix’s catalogue shows a strong shift toward **modern, shorter, and emotionally-driven content**, reflecting global viewer engagement trends.

<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/33c544eb-a90b-4d1a-b184-ef99d0525035" />


---

## Evaluation & Business Impact
- **Elbow & Silhouette Scores** used to find optimal K and assess cluster separation.
- **DBSCAN** identified outlier/noise content, showing sparse niche titles.
- These clusters help Netflix:
  - Curate personalized recommendations.
  - Identify emerging genre trends.
  - Support global content strategy.

---

## Tools & Libraries Used
**Python Libraries:**
`pandas`, `numpy`, `scikit-learn`, `nltk`, `matplotlib`, `seaborn`, `wordcloud`, `contractions`, `scipy`

---

## Files in This Repository
Netflix-Unsupervised-Learning/
│

├── Netflix_unsupervised.ipynb 

├── README.md 

├── requirements.txt 


---

## Future Scope
- Develop an interactive **Streamlit Dashboard** for visual exploration.
- Integrate **Gemini API** for question-answering on Netflix trends.
- Extend to **supervised prediction** (e.g., predict genre cluster or user interest).

---

## Author
**Name:** Kanika Singh Rajpoot

**Project:** Netflix Shows & Movies — Unsupervised Learning Analysis  

**Tools:** Python, scikit-learn, NLTK, Seaborn, WordCloud, Streamlit (optional)  

