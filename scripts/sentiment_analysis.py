"""
sentiment_analysis.py
- Adds sentiment_label and sentiment_score columns using a transformer pipeline if available,
  otherwise falls back to VADER.
- Input: cleaned CSV(s) in in_dir
- Output: updated CSVs saved to out_dir with suffix _sentiment.csv and combined_sentiment.csv
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

def try_transformer_sentiment(texts, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    try:
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline("sentiment-analysis", model=model_name, device=device)
        results = pipe(texts, truncation=True)
        return results
    except Exception as e:
        print("Transformer pipeline not available:", e)
        return None

def vader_sentiment(texts):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    out = []
    for t in texts:
        s = analyzer.polarity_scores(t if isinstance(t,str) else "")
        # map compound to label
        compound = s['compound']
        if compound >= 0.05:
            label = "POSITIVE"
        elif compound <= -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        out.append({"label": label, "score": compound})
    return out

def label_map_transformer(pred):
    # transformer returns {'label': 'POSITIVE', 'score': 0.99}
    return pred

def process_df(df):
    texts = df['review'].fillna("").astype(str).tolist()
    # Try transformer in batches
    batch_size = 32
    results = []
    transformer_results = try_transformer_sentiment(texts[:min(200, len(texts))])
    use_transformer = transformer_results is not None
    if use_transformer:
        print("Using transformer sentiment model.")
        # process in batches
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            try:
                r = try_transformer_sentiment(batch)
                for item in r:
                    results.append({"label": item['label'], "score": float(item['score'])})
            except Exception:
                # fallback on VADER for this batch
                results.extend(vader_sentiment(batch))
    else:
        print("Falling back to VADER sentiment.")
        results = vader_sentiment(texts)

    df['sentiment_label'] = [r['label'] for r in results]
    df['sentiment_score'] = [r['score'] for r in results]
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="data/cleaned")
    parser.add_argument("--out_dir", type=str, default="data/cleaned")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_frames = []
    for csv in in_dir.glob("*_cleaned.csv"):
        print("Running sentiment on", csv)
        df = pd.read_csv(csv)
        df = process_df(df)
        out_path = out_dir / f"{csv.stem.replace('_cleaned','')}_sentiment.csv"
        df.to_csv(out_path, index=False, encoding='utf-8')
        combined_frames.append(df)
    if combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True)
        combined.to_csv(out_dir / "combined_sentiment.csv", index=False, encoding='utf-8')
        print("Wrote combined_sentiment.csv")
    else:
        print("No cleaned CSVs found in", in_dir)

if __name__ == "__main__":
    main()
