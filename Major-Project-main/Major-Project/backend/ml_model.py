"""
CivicPulse Machine Learning Service & Integration Layer (ml_model.py)
====================================================================
Provides a unified, thread-safe, high-performance integration interface for
the CivicPulse backend server, FastAPI routes, and automated worker tasks.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple

from model import (
    CivicMLBundle,
    load_model,
    DEPARTMENTS,
    PRIORITIES,
    SLA_HOURS_MAP
)
from classifier import CivicComplaintClassifier, ClassificationResult

logger = logging.getLogger("civicpulse.ml.service")


class MLModelService:
    """
    Singleton service managing model lifecycle, warm cache inference,
    duplicate cluster detection, and AI analysis for municipal complaints.
    """

    _instance: Optional["MLModelService"] = None

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            # Check local directory or backend directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(base_dir, "model.pkl"),
                os.path.join(base_dir, "Major-Project", "model.pkl"),
                os.path.join(base_dir, "Major-Project", "backend", "model.pkl"),
                os.path.join(base_dir, "..", "model.pkl"),
            ]
            model_path = next((p for p in candidates if os.path.exists(p)), "model.pkl")

        self.model_path = model_path
        self._classifier: Optional[CivicComplaintClassifier] = None
        self._load_classifier()

    @classmethod
    def get_instance(cls, model_path: Optional[str] = None) -> "MLModelService":
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance

    def _load_classifier(self) -> None:
        try:
            if os.path.exists(self.model_path):
                self._classifier = CivicComplaintClassifier(self.model_path)
                logger.info("MLModelService loaded classifier from %s", self.model_path)
            else:
                logger.warning("Model file '%s' not found. Initializing fallback classifier.", self.model_path)
                self._classifier = CivicComplaintClassifier()
        except Exception as e:
            logger.error("Failed to load model from %s: %s. Using rules fallback.", self.model_path, e)
            self._classifier = CivicComplaintClassifier()

    @property
    def classifier(self) -> CivicComplaintClassifier:
        if self._classifier is None:
            self._load_classifier()
        return self._classifier

    def analyze_complaint(
        self,
        title: str,
        description: str,
        ward: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for AI analysis of citizen complaints.
        Combines title + description into a rich representation and returns
        complete prediction results formatted for CivicPulse APIs.
        """
        combined_text = f"{title.strip()} - {description.strip()}".strip(" -")
        if not combined_text:
            combined_text = title or description or "General civic complaint"

        result = self.classifier.classify(combined_text, ward=ward)
        data = result.to_dict()

        # Add top suggestions and UI metadata
        data["title"] = title
        data["ward"] = ward
        data["suggested_departments"] = self._get_top_departments(result.department_probabilities)
        return data

    def detect_duplicates(
        self,
        title: str,
        description: str,
        existing_complaints: List[Dict[str, Any]],
        ward: Optional[str] = None,
        similarity_threshold: float = 0.60
    ) -> List[Dict[str, Any]]:
        """
        Identifies potential duplicate complaints in the same ward using NLP vector similarity.
        """
        if not existing_complaints:
            return []

        # Filter by ward if provided
        filtered = existing_complaints
        if ward:
            filtered = [
                c for c in existing_complaints
                if c.get("ward", "").strip().lower() == ward.strip().lower()
            ] or existing_complaints

        combined_query = f"{title} {description}".strip()

        if self.classifier.bundle and hasattr(self.classifier.bundle, "duplicate_detector"):
            detector = self.classifier.bundle.duplicate_detector
            detector.similarity_threshold = similarity_threshold
            return detector.find_duplicates(combined_query, filtered)

        # Fallback text overlap similarity
        duplicates = []
        q_words = set(combined_query.lower().split())
        for c in filtered:
            c_text = f"{c.get('title', '')} {c.get('description', '')}".lower()
            c_words = set(c_text.split())
            if not q_words or not c_words:
                continue
            jaccard = len(q_words & c_words) / len(q_words | c_words)
            if jaccard >= 0.40:
                match = dict(c)
                match["similarity_score"] = round(jaccard, 3)
                duplicates.append(match)

        duplicates.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        return duplicates

    def batch_analyze(
        self,
        complaints: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyzes a batch of complaint dictionaries.
        Each dict can contain 'title', 'description', and optional 'ward'.
        """
        results = []
        for c in complaints:
            t = c.get("title", "")
            d = c.get("description", "")
            w = c.get("ward", None)
            res = self.analyze_complaint(t, d, ward=w)
            results.append(res)
        return results

    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns model health, version, classes, and training metrics.
        """
        if self.classifier.bundle:
            return {
                "status": "ready",
                "version": self.classifier.bundle.version,
                "trained_at": self.classifier.bundle.trained_at,
                "metrics": self.classifier.bundle.metrics,
                "departments": self.classifier.bundle.departments,
                "priorities": self.classifier.bundle.priorities,
            }
        return {
            "status": "fallback_rules_mode",
            "version": "1.0.0",
            "departments": DEPARTMENTS,
            "priorities": PRIORITIES,
        }

    def _get_top_departments(self, probs: Dict[str, float], top_k: int = 3) -> List[Dict[str, Any]]:
        sorted_depts = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"department": d, "confidence": round(p, 3)} for d, p in sorted_depts]


# Global convenient functions
def get_service(model_path: Optional[str] = None) -> MLModelService:
    return MLModelService.get_instance(model_path)


def ai_analyze_complaint(title: str, description: str, ward: Optional[str] = None) -> Dict[str, Any]:
    return get_service().analyze_complaint(title, description, ward)


def find_duplicates(
    title: str,
    description: str,
    existing: List[Dict[str, Any]],
    ward: Optional[str] = None,
    threshold: float = 0.60
) -> List[Dict[str, Any]]:
    return get_service().detect_duplicates(title, description, existing, ward=ward, similarity_threshold=threshold)
