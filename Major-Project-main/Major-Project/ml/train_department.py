"""
CivicPulse Department Classifier Training Script
================================================
Trains a TF-IDF + Linear SVM model for municipal Department classification
and exports the model as `ml/models/department_model.joblib`.
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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

from preprocessing import CivicTextCleaner


def train_department_model(
    data_path: str = "ml/data/complaints.csv",
    models_dir: str = "ml/models",
    evaluate: bool = True
):
    """
    Trains TF-IDF + Linear SVM for Department classification.
    """
    # Normalize path if run from within ml/
    if not os.path.exists(data_path) and os.path.exists("data/complaints.csv"):
        data_path = "data/complaints.csv"
        models_dir = "models"

    os.makedirs(models_dir, exist_ok=True)
    print(f"[*] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)

    X = df["complaint"].astype(str).tolist()
    y = df["department"].astype(str).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[*] Training Linear SVM Department Classifier on {len(X_train)} samples...")

    pipeline = Pipeline([
        ("cleaner", CivicTextCleaner()),
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=8000,
            strip_accents="unicode"
        )),
        ("svm", SGDClassifier(
            loss="modified_huber",  # Calibrated Linear SVM
            penalty="l2",
            alpha=1e-4,
            max_iter=1000,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    # Save model and standalone vectorizer
    model_path = os.path.join(models_dir, "department_model.joblib")
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")

    joblib.dump(pipeline, model_path)
    joblib.dump(pipeline.named_steps["tfidf"], vectorizer_path)
    print(f"[+] Saved department model to: {model_path}")
    print(f"[+] Saved tfidf vectorizer to: {vectorizer_path}")

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    if evaluate:
        print("\n" + "=" * 60)
        print("  DEPARTMENT CLASSIFIER EVALUATION (TF-IDF + Linear SVM)")
        print("=" * 60)
        print(f"Accuracy: {acc * 100:.2f}% | F1-Score (Weighted): {f1:.4f}\n")
        print(classification_report(y_test, y_pred, digits=4))
        print("=" * 60 + "\n")

    return pipeline, acc, f1


if __name__ == "__main__":
    train_department_model()
