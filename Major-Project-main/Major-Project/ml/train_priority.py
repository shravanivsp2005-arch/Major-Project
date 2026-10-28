"""
CivicPulse Priority Classifier Training Script
==============================================
Trains a Random Forest Classifier using multi-modal features
(TF-IDF textual embeddings + Department + Category + Severity indicators)
to predict issue Priority (LOW, MEDIUM, HIGH, CRITICAL).
Exports the model as `ml/models/priority_model.joblib`.
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

from preprocessing import extract_priority_features, CivicTextCleaner
from pipeline_models import CivicPriorityPipeline


def train_priority_model(
    data_path: str = "ml/data/complaints.csv",
    models_dir: str = "ml/models",
    evaluate: bool = True
):
    """
    Trains Random Forest Classifier for Priority determination.
    """
    if not os.path.exists(data_path) and os.path.exists("data/complaints.csv"):
        data_path = "data/complaints.csv"
        models_dir = "models"

    os.makedirs(models_dir, exist_ok=True)
    print(f"[*] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)

    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    if os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
    else:
        from train_department import train_department_model
        _, _, _ = train_department_model(data_path, models_dir, evaluate=False)
        vectorizer = joblib.load(vectorizer_path)

    X_texts = df["complaint"].astype(str).tolist()
    depts = df["department"].astype(str).tolist()
    cats = df["category"].astype(str).tolist()
    y = df["priority"].astype(str).tolist()

    # Train / Test split
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=y
    )

    X_train_texts = [X_texts[i] for i in train_idx]
    depts_train = [depts[i] for i in train_idx]
    cats_train = [cats[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]

    X_test_texts = [X_texts[i] for i in test_idx]
    depts_test = [depts[i] for i in test_idx]
    cats_test = [cats[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    print(f"[*] Training Random Forest Priority Classifier on {len(train_idx)} samples...")

    # Extract features
    cleaner = CivicTextCleaner()
    X_train_clean = cleaner.transform(X_train_texts)
    X_train_tfidf = vectorizer.transform(X_train_clean)
    X_train_tab = extract_priority_features(X_train_texts, depts_train, cats_train)
    X_train_full = hstack([X_train_tfidf, csr_matrix(X_train_tab)])

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_full, y_train)

    pipeline = CivicPriorityPipeline(vectorizer, rf)

    model_path = os.path.join(models_dir, "priority_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"[+] Saved priority model to: {model_path}")

    # Evaluate
    y_pred = pipeline.predict(X_test_texts, depts_test, cats_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    if evaluate:
        print("\n" + "=" * 60)
        print("  PRIORITY CLASSIFIER EVALUATION (Random Forest)")
        print("=" * 60)
        print(f"Accuracy: {acc * 100:.2f}% | F1-Score (Weighted): {f1:.4f}\n")
        print(classification_report(y_test, y_pred, digits=4))
        print("=" * 60 + "\n")

    return pipeline, acc, f1


if __name__ == "__main__":
    train_priority_model()
