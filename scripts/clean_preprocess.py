"""
clean_preprocess.py
- Reads CSVs in input dir
- Dedupes, normalizes date to YYYY-MM-DD
- Basic cleaning (remove empty reviews)
- Outputs combined cleaned CSV: cleaned_reviews.csv and per-app files in out_dir
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import os
from datetime import datetime

def normalize_date(val):
    if pd.isna(val):
        return None
    # val might be datetime or string
    if isinstance(val, (pd.Timestamp, datetime)):
        return val.strftime("%Y-%m-%d")
    try:
        dt = pd.to_datetime(val)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None

def process_file(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[''])
    # rename columns if necessary
    cols = df.columns.str.lower()
    mapping = {}
    if "content" in cols:
        mapping = {c: c for c in df.columns}  # keep as-is
    # ensure canonical columns
    expected = ["review_id","user","review","rating","date","reply","reply_date","app_id","source"]
    for col in expected:
        if col not in df.columns:
            df[col] = None
    df = df[expected]
    # Convert rating to numeric
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0).astype(int)
    # Normalize date
    df['date'] = df['date'].apply(normalize_date)
    df['reply_date'] = df['reply_date'].apply(normalize_date)
    # Drop rows with no review text and no reply
    df['review'] = df['review'].astype(str).replace({'': None})
    df = df[~(df['review'].isna())].copy()
    # Deduplicate by review_id or review text + app_id
    df = df.drop_duplicates(subset=['review_id'], keep='first')
    df = df.drop_duplicates(subset=['review','app_id'], keep='first')
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="data/raw")
    parser.add_argument("--out_dir", type=str, default="data/cleaned")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for csv in in_dir.glob("*_reviews.csv"):
        print("Processing", csv)
        df = process_file(csv)
        app_id = csv.stem.replace("_reviews","")
        df.to_csv(out_dir / f"{app_id}_cleaned.csv", index=False, encoding='utf-8')
        all_dfs.append(df)
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(out_dir / "cleaned_reviews.csv", index=False, encoding='utf-8')
        print("Wrote combined cleaned_reviews.csv with", len(combined), "rows")
    else:
        print("No review files found in", in_dir)

if __name__ == "__main__":
    main()
