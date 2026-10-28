"""
CivicPulse ML Model Wrapper Classes & In-Memory Pipeline Containers
===================================================================
Provides robust, serializable wrappers for:
- Random Forest Priority Classifier
- SBERT + XGBoost Duplicate Detection Pipeline
"""

import os
import sys
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from geo.haversine import haversine_distance_meters, compute_time_difference_hours
from preprocessing import CivicTextCleaner, extract_priority_features

try:
    from sbert_embedder import get_sbert_embedder
except Exception:
    try:
        from ml.sbert_embedder import get_sbert_embedder
    except Exception:
        get_sbert_embedder = None


class CivicPriorityPipeline:
    """
    Wraps TF-IDF vectorizer + tabular feature extractor + Random Forest model.
    """

    def __init__(self, vectorizer, rf_model):
        self.vectorizer = vectorizer
        self.rf_model = rf_model
        self.classes_ = rf_model.classes_

    def predict(self, X_texts, departments=None, categories=None):
        X_mat = self._build_features(X_texts, departments, categories)
        return self.rf_model.predict(X_mat)

    def predict_proba(self, X_texts, departments=None, categories=None):
        X_mat = self._build_features(X_texts, departments, categories)
        return self.rf_model.predict_proba(X_mat)

    def _build_features(self, X_texts, departments=None, categories=None):
        cleaner = CivicTextCleaner()
        clean_texts = cleaner.transform(X_texts)
        tfidf_feat = self.vectorizer.transform(clean_texts)
        tab_feat = extract_priority_features(X_texts, departments, categories)
        return hstack([tfidf_feat, csr_matrix(tab_feat)])


class CivicDuplicateMLPipeline:
    """
    Hybrid SBERT + TF-IDF + XGBoost Duplicate Detection Pipeline.
    Combines dense semantic sentence embeddings (SBERT), sparse lexical similarity (TF-IDF),
    Haversine geodesic distance, and temporal windowing into an XGBoost classifier.
    """

    def __init__(self, vectorizer, xgb_model):
        self.vectorizer = vectorizer
        self.xgb_model = xgb_model

    def extract_pair_features(self, c1: dict, c2: dict) -> np.ndarray:
        """
        Extracts multi-modal feature vector:
        [sbert_sim, tfidf_sim, haversine_dist_m, time_diff_hrs, cat_match, dept_match]
        """
        t1 = c1.get("complaint") or c1.get("description") or c1.get("title", "")
        t2 = c2.get("complaint") or c2.get("description") or c2.get("title", "")

        # 1. SBERT dense semantic cosine similarity
        sbert_sim = 0.0
        if get_sbert_embedder:
            try:
                sbert = get_sbert_embedder()
                sbert_sim = sbert.cosine_similarity(t1, t2)
            except Exception:
                sbert_sim = 0.0

        # 2. TF-IDF lexical cosine similarity
        cleaner = CivicTextCleaner()
        v1 = self.vectorizer.transform(cleaner.transform([t1]))
        v2 = self.vectorizer.transform(cleaner.transform([t2]))
        tfidf_sim = float(cosine_similarity(v1, v2)[0][0])

        if sbert_sim == 0.0:
            sbert_sim = tfidf_sim

        # 3. Haversine distance in meters
        lat1, lng1 = float(c1.get("latitude", 12.97)), float(c1.get("longitude", 77.59))
        lat2, lng2 = float(c2.get("latitude", 12.97)), float(c2.get("longitude", 77.59))
        dist_m = haversine_distance_meters(lat1, lng1, lat2, lng2)

        # 4. Temporal difference in hours
        time1 = c1.get("timestamp") or c1.get("created_at") or "2026-01-01T00:00:00Z"
        time2 = c2.get("timestamp") or c2.get("created_at") or "2026-01-01T00:00:00Z"
        time_diff = compute_time_difference_hours(time1, time2)

        # 5. Categorical & Department matches
        cat1 = (c1.get("category") or "").strip().lower()
        cat2 = (c2.get("category") or "").strip().lower()
        cat_match = 1.0 if (cat1 and cat1 == cat2) else 0.0

        dept1 = (c1.get("department") or "").strip().lower()
        dept2 = (c2.get("department") or "").strip().lower()
        dept_match = 1.0 if (dept1 and dept1 == dept2) else 0.0

        return np.array([[sbert_sim, tfidf_sim, dist_m, time_diff, cat_match, dept_match]], dtype=np.float32)

    def predict_duplicate_probability(self, c1: dict, c2: dict) -> float:
        """
        Computes probability (0.0 - 1.0) using XGBoost model.
        """
        feat = self.extract_pair_features(c1, c2)
        if hasattr(self.xgb_model, "predict_proba"):
            proba = self.xgb_model.predict_proba(feat)[0][1]
        else:
            proba = float(self.xgb_model.predict(feat)[0])
        return round(float(proba), 4)

    def find_duplicates(self, query_complaint: dict, candidates: list, threshold: float = 0.60) -> list:
        """
        Finds candidate complaints exceeding duplicate threshold.
        """
        results = []
        for cand in candidates:
            score = self.predict_duplicate_probability(query_complaint, cand)
            if score >= threshold:
                results.append({
                    "id": cand.get("id"),
                    "duplicate_group_id": cand.get("duplicate_group_id"),
                    "title": cand.get("title") or cand.get("complaint", ""),
                    "score": score,
                    "candidate": cand
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
