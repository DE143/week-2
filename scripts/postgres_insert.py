"""
postgres_insert.py
- Inserts cleaned & processed reviews into PostgreSQL using SQLAlchemy.
- Creates two tables: banks, reviews (if not exist)
- Requires .env in project root with PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE

Usage:
python scripts/postgres_insert.py --in_dir data/cleaned
"""
import argparse
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

def get_engine():
    host = os.environ.get("PGHOST","localhost")
    port = os.environ.get("PGPORT","5432")
    user = os.environ.get("PGUSER","postgres")
    pw = os.environ.get("PGPASSWORD","postgres")
    db = os.environ.get("PGDATABASE","bank_reviews")
    uri = f"postgresql://{user}:{pw}@{host}:{port}/{db}"
    engine = create_engine(uri)
    return engine

def create_schema(engine):
    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS banks (
            bank_id SERIAL PRIMARY KEY,
            app_id TEXT UNIQUE,
            bank_name TEXT,
            app_name TEXT
        );
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id TEXT PRIMARY KEY,
            bank_id INTEGER REFERENCES banks(bank_id),
            username TEXT,
            review_text TEXT,
            rating INTEGER,
            review_date DATE,
            sentiment_label TEXT,
            sentiment_score FLOAT,
            themes TEXT,
            top_keywords TEXT,
            source TEXT
        );
        """))
        conn.commit()

def upsert_bank(engine, app_id, bank_name=None, app_name=None):
    with engine.connect() as conn:
        # insert or ignore
        res = conn.execute(text("""
        INSERT INTO banks (app_id, bank_name, app_name)
        VALUES (:app_id, :bank_name, :app_name)
        ON CONFLICT (app_id) DO UPDATE
        SET bank_name = COALESCE(excluded.bank_name, banks.bank_name),
            app_name = COALESCE(excluded.app_name, banks.app_name)
        RETURNING bank_id;
        """), {"app_id": app_id, "bank_name": bank_name, "app_name": app_name})
        row = res.fetchone()
        conn.commit()
        return row[0]

def insert_reviews(engine, df):
    # df must have fields: review_id, user, review, rating, date, sentiment_label, sentiment_score, themes, top_keywords, app_id, source
    with engine.connect() as conn:
        for _, row in df.iterrows():
            app_id = row.get('app_id')
            bank_id = upsert_bank(engine, app_id, bank_name=app_id, app_name=app_id)
            try:
                conn.execute(text("""
                    INSERT INTO reviews (review_id, bank_id, username, review_text, rating, review_date, sentiment_label, sentiment_score, themes, top_keywords, source)
                    VALUES (:review_id, :bank_id, :username, :review_text, :rating, :review_date, :sentiment_label, :sentiment_score, :themes, :top_keywords, :source)
                    ON CONFLICT (review_id) DO NOTHING;
                """), {
                    "review_id": str(row.get('review_id')),
                    "bank_id": bank_id,
                    "username": row.get('user'),
                    "review_text": row.get('review'),
                    "rating": int(row.get('rating')) if not pd.isna(row.get('rating')) else None,
                    "review_date": row.get('date'),
                    "sentiment_label": row.get('sentiment_label'),
                    "sentiment_score": float(row.get('sentiment_score')) if not pd.isna(row.get('sentiment_score')) else None,
                    "themes": row.get('themes'),
                    "top_keywords": row.get('top_keywords'),
                    "source": row.get('source')
                })
            except Exception as e:
                print("Error inserting review:", row.get('review_id'), e)
        conn.commit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="data/cleaned")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    combined = in_dir / "combined_themes.csv"
    if not combined.exists():
        print("combined_themes.csv not found — run theme_analysis first.")
        return
    df = pd.read_csv(combined)
    engine = get_engine()
    create_schema(engine)
    insert_reviews(engine, df)
    print("Done inserting reviews.")

if __name__ == "__main__":
    main()
