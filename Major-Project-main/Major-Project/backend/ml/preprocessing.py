"""
CivicPulse ML Preprocessing & Feature Engineering Pipeline
==========================================================
Trained on ground-truth smart_city_civic_complaints.csv and smart_city_civic_dataset.xlsx.
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin

CIVIC_DEPARTMENTS: List[str] = [
    "Water & Sewage",
    "Waste Management",
    "Traffic",
    "Police",
    "Electricity",
    "Roads & Infrastructure",
    "Parks"
]

CIVIC_CATEGORIES: List[str] = [
    "Water Pipeline Leakage",
    "Drainage Overflow",
    "Sewage Blockage",
    "Garbage Overflow",
    "Illegal Dumping",
    "Missed Waste Collection",
    "Traffic Signal Failure",
    "Traffic Congestion",
    "Road Sign Damage",
    "Public Safety Concern",
    "Illegal Parking",
    "Suspicious Activity",
    "Streetlight Failure",
    "Power Infrastructure Damage",
    "Electrical Hazard",
    "Pothole",
    "Road Damage",
    "Footpath Damage",
    "Park Equipment Damage",
    "Broken Bench",
    "Park Maintenance Issue"
]

CIVIC_PRIORITIES: List[str] = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

DEPARTMENT_ALIASES: Dict[str, str] = {
    "police / public safety": "Police",
    "public safety": "Police",
    "police": "Police",
    "roads": "Roads & Infrastructure",
    "water": "Water & Sewage",
    "waste": "Waste Management",
    "power": "Electricity",
    "lighting": "Electricity"
}

SEVERITY_INDICATORS = {
    "CRITICAL": [
        "live wire", "exposed wire", "electrocution", "gas leak", "collapse",
        "fire", "severe danger", "injury", "sinkhole", "major burst",
        "sparking", "fatal", "toxic", "emergency", "rupture", "contamination"
    ],
    "HIGH": [
        "burst", "flooding", "blackout", "no power", "deep pothole",
        "manhole open", "caving in", "overflowing", "accident", "gridlock",
        "hazard", "blocked", "heavy leak", "damage", "theft", "equipment damage"
    ],
    "MEDIUM": [
        "garbage", "pothole", "signal", "streetlight", "broken",
        "foul smell", "leak", "fallen branch", "uncleaned", "bench",
        "congestion", "failure", "dumping", "suspicious", "maintenance"
    ],
    "LOW": [
        "minor", "cosmetic", "faded", "slow", "dry patch",
        "query", "tree pruning", "faded paint", "missed collection",
        "footpath"
    ]
}


class CivicTextCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans raw complaint text: lowercasing, URL/email/number stripping,
    whitespace normalization, and punctuation cleaning.
    """

    def __init__(self, lower: bool = True, strip_punctuation: bool = True):
        self.lower = lower
        self.strip_punctuation = strip_punctuation

    def fit(self, X, y=None):
        return self

    def transform(self, X: List[str]) -> List[str]:
        cleaned = []
        for text in X:
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            cleaned.append(self.clean_text(text))
        return cleaned

    def clean_text(self, text: str) -> str:
        if self.lower:
            text = text.lower()
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        # Remove emails & phone numbers
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text)
        text = re.sub(r"\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", " ", text)
        if self.strip_punctuation:
            text = re.sub(r"[^\w\s\-\/]", " ", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text


def extract_priority_features(
    texts: List[str],
    departments: Optional[List[str]] = None,
    categories: Optional[List[str]] = None
) -> np.ndarray:
    """
    Extracts tabular numerical / categorical feature vectors for the Random Forest Priority model.
    """
    cleaner = CivicTextCleaner()
    cleaned_texts = cleaner.transform(texts)
    num_samples = len(texts)

    dept_list = departments or ["" for _ in range(num_samples)]
    cat_list = categories or ["" for _ in range(num_samples)]

    features = []
    for i, t in enumerate(cleaned_texts):
        feat_vec = []
        char_len = len(t)
        words = t.split()
        word_count = len(words)
        feat_vec.extend([char_len, word_count])

        crit_count = sum(1 for kw in SEVERITY_INDICATORS["CRITICAL"] if kw in t)
        high_count = sum(1 for kw in SEVERITY_INDICATORS["HIGH"] if kw in t)
        med_count = sum(1 for kw in SEVERITY_INDICATORS["MEDIUM"] if kw in t)
        low_count = sum(1 for kw in SEVERITY_INDICATORS["LOW"] if kw in t)
        feat_vec.extend([crit_count, high_count, med_count, low_count])

        cur_dept = dept_list[i] if i < len(dept_list) else ""
        cur_dept = DEPARTMENT_ALIASES.get(cur_dept.lower(), cur_dept)
        dept_onehot = [1.0 if cur_dept == d else 0.0 for d in CIVIC_DEPARTMENTS]
        feat_vec.extend(dept_onehot)

        cur_cat = cat_list[i] if i < len(cat_list) else ""
        cat_onehot = [1.0 if cur_cat == c else 0.0 for c in CIVIC_CATEGORIES]
        feat_vec.extend(cat_onehot)

        features.append(feat_vec)

    return np.array(features, dtype=np.float32)
