"""
CivicPulse XGBoost + SBERT Duplicate Complaint Model Trainer
============================================================
Trains an XGBoost (Extreme Gradient Boosting) model on synthesized complaint pairs
using SBERT dense contextual embeddings, TF-IDF lexical similarity, Haversine
geodesic distance, temporal delta, and category matching.
"""

import os
import sys
import random
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False
    from sklearn.ensemble import GradientBoostingClassifier

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from geo.haversine import haversine_distance_meters
from preprocessing import CivicTextCleaner
from pipeline_models import CivicDuplicateMLPipeline

try:
    from sbert_embedder import get_sbert_embedder
except Exception:
    from ml.sbert_embedder import get_sbert_embedder


def synthesize_pairwise_dataset(
    df: pd.DataFrame,
    vectorizer,
    num_pairs: int = 4000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesizes positive (duplicate) and negative (distinct) complaint pairs with:
    [sbert_sim, tfidf_sim, haversine_dist_m, time_diff_hrs, cat_match, dept_match]
    """
    cleaner = CivicTextCleaner()
    cleaned_texts = cleaner.transform(df["complaint"].tolist())
    tfidf_matrix = vectorizer.transform(cleaned_texts)

    sbert = get_sbert_embedder()
    sbert_vecs = sbert.encode(df["complaint"].tolist(), normalize=True)

    X_feats = []
    y_labels = []

    records = df.to_dict("records")
    n = len(records)

    # 1. Positive Pairs (Duplicates)
    pos_count = num_pairs // 2
    for _ in range(pos_count):
        idx1 = random.randint(0, n - 1)
        r1 = records[idx1]

        # Duplicate in close proximity (<200m) and temporal window (<72h)
        same_cat_indices = [i for i, r in enumerate(records) if r["category"] == r1["category"]]
        idx2 = random.choice(same_cat_indices) if same_cat_indices else idx1
        r2 = records[idx2]

        sbert_sim = float(np.dot(sbert_vecs[idx1], sbert_vecs[idx2]))
        tfidf_sim = float(tfidf_matrix[idx1].dot(tfidf_matrix[idx2].T).toarray()[0][0])
        dist_m = random.uniform(5.0, 180.0)  # within 200m
        time_diff = random.uniform(0.5, 48.0)  # within 72h
        cat_match = 1.0
        dept_match = 1.0

        X_feats.append([sbert_sim, tfidf_sim, dist_m, time_diff, cat_match, dept_match])
        y_labels.append(1)

    # 2. Negative Pairs (Non-Duplicates)
    neg_count = num_pairs - pos_count
    for _ in range(neg_count):
        idx1 = random.randint(0, n - 1)
        idx2 = random.randint(0, n - 1)
        while idx2 == idx1:
            idx2 = random.randint(0, n - 1)

        r1 = records[idx1]
        r2 = records[idx2]

        sbert_sim = float(np.dot(sbert_vecs[idx1], sbert_vecs[idx2]))
        tfidf_sim = float(tfidf_matrix[idx1].dot(tfidf_matrix[idx2].T).toarray()[0][0])
        dist_m = random.uniform(500.0, 15000.0)  # > 500m
        time_diff = random.uniform(1.0, 500.0)
        cat_match = 1.0 if r1["category"] == r2["category"] else 0.0
        dept_match = 1.0 if r1["department"] == r2["department"] else 0.0

        X_feats.append([sbert_sim, tfidf_sim, dist_m, time_diff, cat_match, dept_match])
        y_labels.append(0)

    return np.array(X_feats, dtype=np.float32), np.array(y_labels, dtype=np.int32)


def train_duplicate_model(
    data_path: str = "ml/data/complaints.csv",
    models_dir: str = "ml/models",
    evaluate: bool = True
) -> Tuple[CivicDuplicateMLPipeline, float, float]:
    """
    Trains and serializes the XGBoost duplicate detection pipeline.
    """
    os.makedirs(models_dir, exist_ok=True)
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    if not os.path.exists(vec_path):
        raise FileNotFoundError(f"TF-IDF vectorizer not found at '{vec_path}'.")

    vectorizer = joblib.load(vec_path)

    print(f"[*] Loading dataset from {data_path} for SBERT + XGBoost duplicate pair synthesis...")
    df = pd.read_csv(data_path)

    X, y = synthesize_pairwise_dataset(df, vectorizer, num_pairs=4000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    if XGB_AVAILABLE:
        print(f"[*] Training XGBoost Classifier (XGBClassifier) on {len(X_train)} complaint pairs...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            eval_metric="logloss",
            random_state=42
        )
    else:
        print(f"[*] Training Gradient Boosting Duplicate Classifier on {len(X_train)} pairs...")
        xgb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            random_state=42
        )

    xgb_model.fit(X_train, y_train)

    pipeline = CivicDuplicateMLPipeline(vectorizer, xgb_model)

    y_pred = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    if evaluate:
        print(f"\n[+] XGBoost Duplicate Model Evaluation:")
        print(f"    Accuracy: {acc*100:.2f}% | F1-Score: {f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Non-Duplicate", "Duplicate"], digits=4))

    save_path = os.path.join(models_dir, "duplicate_model.joblib")
    joblib.dump(pipeline, save_path)
    print(f"[+] Saved XGBoost duplicate detection model to: {save_path}")

    return pipeline, acc, f1


if __name__ == "__main__":
    train_duplicate_model()
