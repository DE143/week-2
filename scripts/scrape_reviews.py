"""
scrape_reviews.py
Scrapes reviews from Google Play using google-play-scraper.

Usage:
python scripts/scrape_reviews.py --apps "com.bank.app1,com.bank.app2" --out_dir data/raw --min_reviews 400

Output: CSV per app saved to out_dir/<app_id>_reviews.csv
"""
import argparse
import csv
import time
from pathlib import Path
from tqdm import tqdm
from google_play_scraper import Sort, reviews_all

def scrape_app(app_id, min_reviews=400):
    # reviews_all returns a list of dicts
    print(f"Scraping {app_id} ... requesting {min_reviews} reviews (may take a while).")
    result = reviews_all(
        app_id,
        sleep_milliseconds=0, # be gentle; we will add delays
        lang='en',
        country='us',
        sort=Sort.NEWEST
    )
    print(f"Got {len(result)} reviews for {app_id}")
    return result

def save_reviews(reviews, out_path, app_id):
    # normalize fields
    keys = ["reviewId", "userName", "content", "score", "at", "replyText", "replyDate"]
    with open(out_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["review_id","user","review","rating","date","reply","reply_date","app_id","source"])
        for r in reviews:
            row = [
                r.get("reviewId"),
                r.get("userName"),
                r.get("content"),
                r.get("score"),
                r.get("at"),
                r.get("replyText"),
                r.get("replyDate"),
                app_id,
                "google_play"
            ]
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apps", type=str, required=True,
                        help="Comma-separated app IDs (play store package names)")
    parser.add_argument("--out_dir", type=str, default="data/raw")
    parser.add_argument("--min_reviews", type=int, default=400)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    app_ids = [a.strip() for a in args.apps.split(",") if a.strip()]
    for app_id in app_ids:
        reviews = scrape_app(app_id, args.min_reviews)
        # if fewer than min_reviews, warn
        if len(reviews) < args.min_reviews:
            print(f"Warning: {app_id} returned only {len(reviews)} reviews (min requested {args.min_reviews})")
        out_path = out_dir / f"{app_id}_reviews.csv"
        save_reviews(reviews, out_path, app_id)
        time.sleep(2)

if __name__ == "__main__":
    main()
