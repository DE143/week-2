"""
visualization.py
- Produces key plots: rating distribution, sentiment counts per bank, top keywords wordcloud, themes bar chart.
- Saves PNGs to out_dir (e.g., reports/)
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

def plot_rating_distribution(df, out_path):
    plt.figure(figsize=(6,4))
    df['rating'].value_counts().sort_index().plot(kind='bar')
    plt.xlabel("Rating (stars)")
    plt.ylabel("Count")
    plt.title("Overall Rating Distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_sentiment_by_bank(df, out_path):
    pivot = pd.crosstab(df['app_id'], df['sentiment_label'])
    pivot.plot(kind='bar', stacked=True, figsize=(8,5))
    plt.ylabel("Count")
    plt.title("Sentiment by App")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def wordcloud_from_keywords(df, out_path):
    all_kws = " ".join(df['top_keywords'].fillna("").astype(str).str.replace(";", " "))
    wc = WordCloud(width=800, height=400).generate(all_kws)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def theme_bar_chart(df, out_path):
    # split themes and count
    s = df['themes'].fillna("Other").astype(str)
    all_themes = s.str.split(",").explode()
    counts = all_themes.value_counts().head(20)
    plt.figure(figsize=(8,6))
    counts.plot(kind='barh')
    plt.xlabel("Count")
    plt.title("Top Themes (overall)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="data/cleaned")
    parser.add_argument("--out_dir", type=str, default="reports")
    args = parser.parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = in_dir / "combined_themes.csv"
    if not combined.exists():
        print("combined_themes.csv not found.")
        return
    df = pd.read_csv(combined)

    plot_rating_distribution(df, out_dir / "rating_distribution.png")
    print("Saved rating_distribution.png")
    plot_sentiment_by_bank(df, out_dir / "sentiment_by_app.png")
    print("Saved sentiment_by_app.png")
    wordcloud_from_keywords(df, out_dir / "keywords_wordcloud.png")
    print("Saved keywords_wordcloud.png")
    theme_bar_chart(df, out_dir / "themes_bar.png")
    print("Saved themes_bar.png")

if __name__ == "__main__":
    main()
