"""
CivicPulse Machine Learning Core Models and Pipeline Architectures
===================================================================
Defines the text preprocessing pipelines, feature extractors, multi-target
classifiers (Department & Priority), duplicate detection engine, and model
serialization bundle for smart city civic complaint routing.
"""

import os
import re
import math
import pickle
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("civicpulse.ml.model")

# Standard Domain Constants for Smart City Civic Complaints
DEPARTMENTS: List[str] = [
    "Water & Sewage",
    "Waste Management",
    "Traffic",
    "Police / Public Safety",
    "Electricity",
    "Roads & Infrastructure",
    "Parks",
]

PRIORITIES: List[str] = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

SLA_HOURS_MAP: Dict[str, int] = {
    "CRITICAL": 6,
    "HIGH": 24,
    "MEDIUM": 72,
    "LOW": 168,
}

DEPARTMENT_KEYWORDS: Dict[str, List[str]] = {
    "Water & Sewage": [
        "water", "pipe", "pipeline", "leak", "leakage", "sewage", "drain", "drainage",
        "tap", "overflow", "flooding", "manhole", "contamination", "drinking water",
        "gutter", "borewell", "waterlogging", "sewer", "clogged drain", "burst pipe"
    ],
    "Waste Management": [
        "garbage", "trash", "waste", "litter", "dump", "dustbin", "rubbish", "debris",
        "sanitation", "smell", "rotting", "refuse", "compost", "uncollected waste",
        "solid waste", "plastic dump", "cleanliness", "sweep"
    ],
    "Traffic": [
        "traffic", "signal", "traffic light", "congestion", "jam", "parking", "vehicle",
        "road block", "zebra crossing", "junction", "one way", "illegal parking",
        "speed breaker", "blind spot", "traffic police", "gridlock", "bottleneck"
    ],
    "Police / Public Safety": [
        "theft", "safety", "crime", "harassment", "assault", "suspicious", "accident",
        "violence", "unsafe", "vandalism", "robbery", "eve teasing", "security",
        "emergency", "law and order", "brawl", "stalking", "unattended bag"
    ],
    "Electricity": [
        "electric", "power", "wire", "cable", "transformer", "streetlight", "street light",
        "outage", "shock", "pole", "voltage", "blackout", "spark", "loose wire",
        "hanging wire", "meter", "load shedding", "no electricity", "short circuit"
    ],
    "Roads & Infrastructure": [
        "road", "pothole", "footpath", "sidewalk", "bridge", "crack", "construction",
        "pavement", "flyover", "asphalt", "tar road", "caved road", "uneven road",
        "damaged divider", "barricade", "excavation", "broken curb", "street cave-in"
    ],
    "Parks": [
        "park", "tree", "garden", "playground", "bench", "lawn", "greenery", "plant",
        "fallen branch", "overgrown weed", "walking track", "swing", "jogging path",
        "dead tree", "fence", "sprinkler", "public park"
    ]
}

SEVERITY_KEYWORDS: Dict[str, List[str]] = {
    "CRITICAL": [
        "live wire", "exposed wire", "electrocution", "gas leak", "collapse",
        "fire", "severe danger", "injury", "sinkhole", "major burst",
        "sewage overflow into house", "drinking water contaminated", "fatal", "sparking wire"
    ],
    "HIGH": [
        "major leak", "pipe burst", "head-on accident", "blocked road", "flooding",
        "complete blackout", "no power", "deep pothole", "broken transformer",
        "manhole open", "caving in", "overflowing sewage"
    ],
    "MEDIUM": [
        "garbage pile", "pothole", "signal not working", "streetlight dim",
        "water pressure low", "illegal parking", "fallen branch", "uncleaned bin"
    ],
    "LOW": [
        "minor litter", "bench paint peeling", "faded road marking", "slow tap",
        "dry garden patch", "cosmetic crack", "general query", "tree pruning needed"
    ]
}


class CivicTextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Cleans, normalizes, and prepares civic complaint text for NLP processing.
    """

    def __init__(self, lower: bool = True, remove_urls: bool = True, clean_numbers: bool = False):
        self.lower = lower
        self.remove_urls = remove_urls
        self.clean_numbers = clean_numbers

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[str]) -> List[str]:
        cleaned = []
        for text in X:
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            cleaned.append(self._clean_text(text))
        return cleaned

    def _clean_text(self, text: str) -> str:
        if self.lower:
            text = text.lower()
        if self.remove_urls:
            text = re.sub(r"https?://\S+|www\.\S+", "", text)
        # Remove emails and phone numbers
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text)
        text = re.sub(r"\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", " ", text)
        if self.clean_numbers:
            text = re.sub(r"\b\d+\b", " ", text)
        # Normalize punctuation and extra spaces
        text = re.sub(r"[^\w\s\-\/]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


class CivicDepartmentModel:
    """
    NLP Pipeline for Municipal Department Classification.
    Uses TF-IDF unigram + bigram representation combined with Complement Naive Bayes.
    """

    def __init__(self, max_features: int = 5000, alpha: float = 0.1):
        self.max_features = max_features
        self.alpha = alpha
        self.pipeline = Pipeline([
            ("preprocessor", CivicTextPreprocessor()),
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                max_features=self.max_features,
                min_df=1,
                strip_accents="unicode"
            )),
            ("clf", ComplementNB(alpha=self.alpha, norm=True))
        ])
        self.is_fitted = False

    def fit(self, X: List[str], y: List[str]):
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: List[str]) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        return self.pipeline.named_steps["clf"].classes_


class CivicPriorityModel:
    """
    NLP Pipeline for Complaint Severity / Priority Classification (LOW, MEDIUM, HIGH, CRITICAL).
    Uses TF-IDF representation combined with Complement Naive Bayes.
    """

    def __init__(self, max_features: int = 4000, alpha: float = 0.2):
        self.max_features = max_features
        self.alpha = alpha
        self.pipeline = Pipeline([
            ("preprocessor", CivicTextPreprocessor()),
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                max_features=self.max_features,
                min_df=1,
                strip_accents="unicode"
            )),
            ("clf", ComplementNB(alpha=self.alpha, norm=True))
        ])
        self.is_fitted = False

    def fit(self, X: List[str], y: List[str]):
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: List[str]) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        return self.pipeline.named_steps["clf"].classes_


class CivicDuplicateDetector:
    """
    TF-IDF Vectorizer and similarity search engine to detect duplicate or near-identical
    complaints submitted within the same ward or geographical vicinity.
    """

    def __init__(self, similarity_threshold: float = 0.65):
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=3000,
            strip_accents="unicode"
        )
        self.is_fitted = False

    def fit(self, corpus: List[str]):
        if len(corpus) > 0:
            self.vectorizer.fit(corpus)
            self.is_fitted = True
        return self

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        if not self.is_fitted:
            self.fit([text_a, text_b])
        vecs = self.vectorizer.transform([text_a, text_b])
        sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
        return float(sim)

    def find_duplicates(self, query_text: str, candidate_complaints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compares query_text against a list of existing complaints and returns matches exceeding threshold.
        """
        if not candidate_complaints or not query_text:
            return []

        if not self.is_fitted:
            texts = [c.get("title", "") + " " + c.get("description", "") for c in candidate_complaints] + [query_text]
            self.fit(texts)

        query_vec = self.vectorizer.transform([query_text])
        candidate_texts = [
            (c.get("title", "") + " " + c.get("description", "")).strip()
            for c in candidate_complaints
        ]
        cand_vecs = self.vectorizer.transform(candidate_texts)
        sims = cosine_similarity(query_vec, cand_vecs)[0]

        duplicates = []
        for idx, sim in enumerate(sims):
            if sim >= self.similarity_threshold:
                match = dict(candidate_complaints[idx])
                match["similarity_score"] = round(float(sim), 4)
                duplicates.append(match)

        duplicates.sort(key=lambda x: x["similarity_score"], reverse=True)
        return duplicates


@dataclass
class CivicMLBundle:
    """
    Serializable bundle container storing all trained models, metadata, mappings, and metrics.
    """
    department_model: CivicDepartmentModel
    priority_model: CivicPriorityModel
    duplicate_detector: CivicDuplicateDetector
    departments: List[str] = field(default_factory=lambda: list(DEPARTMENTS))
    priorities: List[str] = field(default_factory=lambda: list(PRIORITIES))
    sla_hours: Dict[str, int] = field(default_factory=lambda: dict(SLA_HOURS_MAP))
    department_keywords: Dict[str, List[str]] = field(default_factory=lambda: dict(DEPARTMENT_KEYWORDS))
    severity_keywords: Dict[str, List[str]] = field(default_factory=lambda: dict(SEVERITY_KEYWORDS))
    metrics: Dict[str, Any] = field(default_factory=dict)
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "2.0.0"

    def save(self, file_path: str) -> None:
        """Serializes bundle to a .pkl file."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved CivicPulse ML Bundle to %s", file_path)

    @classmethod
    def load(cls, file_path: str) -> "CivicMLBundle":
        """Loads serialized bundle from a .pkl file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model bundle not found at: {file_path}")
        with open(file_path, "rb") as f:
            bundle = pickle.load(f)
        logger.info("Loaded CivicPulse ML Bundle from %s (version: %s)", file_path, bundle.version)
        return bundle


def save_model(bundle: CivicMLBundle, path: str = "model.pkl") -> None:
    """Convenience helper to save bundle to path."""
    bundle.save(path)


def load_model(path: str = "model.pkl") -> CivicMLBundle:
    """Convenience helper to load bundle from path."""
    return CivicMLBundle.load(path)
