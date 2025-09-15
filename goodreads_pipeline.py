#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project: Goodreads Recommendation System
Purpose: End-to-end pipeline to clean data, train baseline + collaborative filtering (CF) models, evaluate them, and export top-N recommendations.
Output: One clear, reproducible entry point that runs locally and in CI, with readable comments.
"""

from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

# We use Surprise because it provides well-tested CF algorithms (KNN, SVD) with simple APIs,
# stable train/test utilities, and consistent prediction objects—great for quick, reliable baselines.
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from collections import defaultdict


# Configuration objects keep paths and settings organized and easy to override from CLI.
@dataclass
class Paths:
    books_csv: str = "data/books.csv"
    ratings_csv: str = "data/ratings.csv"
    out_dir: str = "artifacts"

@dataclass
class Settings:
    # Group model/eval knobs here so experiments are easier to track and tweak.
    min_ratings_per_user: int = 100        # Why 100: helps CF by ensuring users have enough history to be comparable
    test_size: float = 0.2                 # Standard 80/20 split keeps results comparable across runs
    random_state: int = 20057              # Fixed seed for reproducibility in reports/CI. I like to use the GU zip code 
    k_recs: int = 10                       # Top-N size aligns with common recsys reporting
    sim_name: str = "cosine"               # Cosine is a robust default for sparse, bounded ratings
    user_based: bool = True                # True = UserKNN, False = ItemKNN; lets us compare taste vs. co-consumption


# Simple I/O helpers
def ensure_dir(path: str) -> None:
    # Create directory if missing so first runs don't crash
    os.makedirs(path, exist_ok=True)

def save_json(obj: Dict[str, Any], path: str) -> None:
    # Persist small results (metrics, EDA summaries) so the report can read them later
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# Data loading and basic cleaning
def load_raw(paths: Paths) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Load raw CSVs; keep loading logic minimal and explicit
    books = pd.read_csv(paths.books_csv)
    ratings = pd.read_csv(paths.ratings_csv)
    return books, ratings

def clean_books(books: pd.DataFrame) -> pd.DataFrame:
    # Drop duplicate titles so a single work doesn't appear multiple times and skew similarity
    before = len(books)
    books = books.drop_duplicates(subset=["title"], keep="first").copy()
    after = len(books)
    print(f"[clean_books] Kept {after}/{before} unique titles")
    return books

def clean_ratings(ratings: pd.DataFrame, valid_book_ids: set, min_ratings: int) -> pd.DataFrame:
    # Remove exact duplicate (user, book, rating) rows to avoid double-counting signals
    # Filter to valid book_ids to keep the matrix aligned with the cleaned catalog
    # Keep only engaged users (>= min_ratings) because CF relies on overlap across users/items
    before = len(ratings)
    ratings = ratings.drop_duplicates(subset=["user_id", "book_id", "rating"], keep="first")
    ratings = ratings[ratings["book_id"].isin(valid_book_ids)]

    user_counts = ratings.groupby("user_id")["book_id"].count()
    keep_users = user_counts[user_counts >= min_ratings].index
    ratings = ratings[ratings["user_id"].isin(keep_users)].copy()

    after = len(ratings)
    print(f"[clean_ratings] Kept {after}/{before} ratings after dedupe/filtering")
    return ratings


# Minimal EDA for quick sanity checks and for inclusion in the report
def quick_eda(ratings: pd.DataFrame) -> Dict[str, Any]:
    # Describe the rating distribution and histogram; helps detect degenerate data (e.g., all 5s)
    desc = ratings["rating"].describe().to_dict()
    hist = ratings["rating"].value_counts().sort_index().to_dict()
    return {"rating_describe": desc, "rating_hist": hist}


# Prepare Surprise dataset
def to_surprise(ratings: pd.DataFrame) -> Dataset:
    # Map to Surprise's expected (user, item, rating) format with an explicit rating scale
    # Using the observed min/max keeps things consistent across different datasets.
    reader = Reader(rating_scale=(ratings["rating"].min(), ratings["rating"].max()))
    data = Dataset.load_from_df(ratings[["user_id", "book_id", "rating"]], reader)
    return data


# Model training and evaluation
def train_knn(data: Dataset, settings: Settings):
    # Single split keeps experiments fast and reproducible; for production use CV.
    trainset, testset = train_test_split(data, test_size=settings.test_size, random_state=settings.random_state)

    # KNNBasic is a transparent baseline (easy to explain/debug).
    # sim_options lets us toggle user-based (taste similarity) vs item-based (co-consumption).
    sim_options = {"name": settings.sim_name, "user_based": settings.user_based}
    algo = KNNBasic(sim_options=sim_options, verbose=False)

    algo.fit(trainset)
    predictions = algo.test(testset)
    return algo, predictions

def rmse(predictions) -> float:
    # Standard regression-style error for explicit ratings; complements top-N metrics
    se = [(pred.r_ui - pred.est) ** 2 for pred in predictions]
    return float(np.sqrt(np.mean(se)))

def precision_recall_at_k(predictions, k=10, threshold=4.0) -> Tuple[float, float]:
    # Evaluate top-N ranking quality: precision@k (how many of top-k are "good") and recall@k
    # threshold=4.0 treats 4–5 ratings as relevant; adjust to match domain expectations.
    user_est_true = defaultdict(list)
    for p in predictions:
        user_est_true[p.uid].append((p.est, p.r_ui))

    precisions, recalls = [], []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]
        n_rel = sum((true >= threshold) for (_, true) in user_ratings)
        n_rec_k = sum((true >= threshold) for (_, true) in top_k)
        precision = n_rec_k / k if k else 0.0
        recall = n_rec_k / n_rel if n_rel else 0.0
        precisions.append(precision)
        recalls.append(recall)
    return float(np.mean(precisions)), float(np.mean(recalls))


# Generating top-N recommendations for a specific user
def recommend_for_user(algo, user_id: Any, all_item_ids: List[Any], known_items: set, k=10) -> List[Any]:
    # Score only unseen items to avoid recommending what the user already interacted with
    # Sorting by predicted rating is simple and aligns with training objective.
    candidates = [iid for iid in all_item_ids if iid not in known_items]
    preds = [(iid, algo.predict(user_id, iid).est) for iid in candidates]
    preds.sort(key=lambda x: x[1], reverse=True)
    return [iid for iid, _ in preds[:k]]


# Main entry point with a small CLI for reproducible experiments
def main():
    parser = argparse.ArgumentParser(description="Goodreads CF pipeline (clean, train, evaluate, recommend)")
    parser.add_argument("--books_csv", default="data/books.csv")
    parser.add_argument("--ratings_csv", default="data/ratings.csv")
    parser.add_argument("--out_dir", default="artifacts")
    parser.add_argument("--user_based", action="store_true", help="Use UserKNN (default is ItemKNN)")
    parser.add_argument("--k_recs", type=int, default=10)
    parser.add_argument("--min_ratings", type=int, default=100)
    args = parser.parse_args()

    # Pack configs so downstream functions remain clean and testable
    paths = Paths(books_csv=args.books_csv, ratings_csv=args.ratings_csv, out_dir=args.out_dir)
    settings = Settings(min_ratings_per_user=args.min_ratings,
                        user_based=args.user_based,
                        k_recs=args.k_recs)

    ensure_dir(paths.out_dir)

    # 1) Load
    books_raw, ratings_raw = load_raw(paths)

    # 2) Clean
    books = clean_books(books_raw)
    ratings = clean_ratings(ratings_raw, valid_book_ids=set(books["book_id"]), min_ratings=settings.min_ratings_per_user)

    # 3) EDA
    eda = quick_eda(ratings)
    save_json(eda, os.path.join(paths.out_dir, "eda_summary.json"))

    # 4) Prepare Surprise dataset
    data = to_surprise(ratings)

    # 5) Train and evaluate
    algo, preds = train_knn(data, settings)
    metrics = {
        "rmse": rmse(preds),
    }
    p_at_k, r_at_k = precision_recall_at_k(preds, k=settings.k_recs)
    metrics.update({"precision@k": p_at_k, "recall@k": r_at_k})
    save_json(metrics, os.path.join(paths.out_dir, "metrics.json"))
    print("[metrics]", metrics)

    # 6) Export example recommendations for one user
    # Picking the first filtered user keeps this step deterministic.
    first_user = int(ratings["user_id"].iloc[0])
    user_known = set(ratings.loc[ratings["user_id"] == first_user, "book_id"])
    item_ids = list(ratings["book_id"].unique())
    topn = recommend_for_user(algo, user_id=first_user, all_item_ids=item_ids, known_items=user_known, k=settings.k_recs)
    save_json({"user_id": first_user, "recommendations": topn}, os.path.join(paths.out_dir, "sample_recs.json"))

    # 7) Done
    print(f"[done] Artifacts saved under: {paths.out_dir}")


if __name__ == "__main__":
    main()
