"""
theme_analysis.py
- Extracts keywords using TF-IDF and clusters or groups keywords into themes.
- Produces theme labels for each review using simple rule-based matching and (optionally) LDA.
- Writes: combined_themes.csv with columns: review_id, review, themes (comma-separated), top_keywords
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from tqdm import tqdm
import spacy
from collections import Counter

nlp = None
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    # will lazy-load if needed
    pass

THEME_KEYWORDS = {
    "Account Access": ["login", "signin", "password", "otp", "forgot", "biometric", "fingerprint", "pin", "auth"],
    "Transaction Performance": ["slow", "timeout", "transfer", "processing", "lag", "speed", "transfered", "failed"],
    "Crashes & Bugs": ["crash", "bug", "error", "exception", "freeze", "hang"],
    "User Interface": ["ui", "user interface", "design", "layout", "button", "menu", "navigation"],
    "Customer Support": ["support", "customer service", "call", "agent", "response", "ticket"],
    "Feature Requests": ["feature", "add", "request", "notification", "fingerprint", "multi-step", "export"]
}

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+',' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_keywords_tfidf(corpus, top_n=5):
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    top_keywords = []
    for row in X:
        if hasattr(row, "toarray"):
            arr = row.toarray()[0]
        else:
            arr = row
        top_idx = arr.argsort()[-top_n:][::-1]
        kws = [feature_names[i] for i in top_idx if arr[i] > 0]
        top_keywords.append(kws)
    return top_keywords

def rule_based_themes(text):
    themes = []
    t = text.lower()
    for theme, keywords in THEME_KEYWORDS.items():
        for k in keywords:
            if k in t:
                themes.append(theme)
                break
    if not themes:
        themes.append("Other")
    return themes

def nmf_topics(corpus, n_topics=6, n_top_words=6):
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(corpus)
    nmf = NMF(n_components=n_topics, random_state=42)
    W = nmf.fit_transform(X)
    H = nmf.components_
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(H):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        topics.append([feature_names[i] for i in top_indices])
    return topics, W

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="data/cleaned")
    parser.add_argument("--out_dir", type=str, default="data/cleaned")
    parser.add_argument("--n_topics", type=int, default=6)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_path = in_dir / "combined_sentiment.csv"
    if not combined_path.exists():
        print("Missing combined_sentiment.csv. Run sentiment_analysis first.")
        return
    df = pd.read_csv(combined_path)
    df['clean_text'] = df['review'].fillna("").astype(str).map(preprocess_text)

    # TF-IDF top keywords (fast)
    print("Extracting TF-IDF keywords...")
    tfidf_keywords = extract_keywords_tfidf(df['clean_text'].tolist(), top_n=6)
    df['top_keywords'] = [";".join(k) for k in tfidf_keywords]

    # Rule-based theme labelling
    print("Applying rule-based theme mapping...")
    df['themes'] = df['clean_text'].apply(lambda t: ",".join(rule_based_themes(t)))

    # Optional: NMF topics to surface global topics
    print("Running NMF topic model (optional)...")
    topics, W = nmf_topics(df['clean_text'].tolist(), n_topics=args.n_topics)
    print("Identified NMF topics (example words):")
    for i, t in enumerate(topics):
        print(f"Topic {i}: {', '.join(t)}")

    # Save
    df_out = df[['review_id','user','review','rating','date','app_id','source','sentiment_label','sentiment_score','top_keywords','themes']]
    df_out.to_csv(out_dir / "combined_themes.csv", index=False, encoding='utf-8')
    # also write per-app
    for app, g in df_out.groupby('app_id'):
        g.to_csv(out_dir / f"{app}_themes.csv", index=False, encoding='utf-8')
    print("Wrote combined_themes.csv and per-app theme CSVs")

if __name__ == "__main__":
    main()
