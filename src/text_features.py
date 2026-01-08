import re
import string
import nltk
import contractions
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_and_lemmatize(text: str) -> str:
    if not isinstance(text, str) or text.strip() == '':
        return ''

    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\w*\d\w*', '', text)

    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 1
    ]
    return ' '.join(tokens)


def build_text_embeddings(df: pd.DataFrame):
    df = df.copy()
    df['clean_description'] = df['description'].apply(clean_and_lemmatize)


    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words='english'
    )

    tfidf_matrix = vectorizer.fit_transform(df['clean_description'])

    svd = TruncatedSVD(n_components=50, random_state=42)
    X_svd = svd.fit_transform(tfidf_matrix)

    return X_svd, vectorizer, svd
