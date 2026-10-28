"""
CivicPulse Master Prediction Engine (ml/predict.py)
===================================================
Production inference engine orchestrating all models:
1. TF-IDF + Linear SVM (Department & Category Routing)
2. Random Forest (Priority & Urgency SLA Scorer)
3. Ultralytics YOLOv8 (Computer Vision Defect Localizer)
4. SBERT + XGBoost (Pairwise Duplicate Incident Detector)
5. Local LLM / XAI (Natural Language Reasoning Engine)
"""

import os
import sys
import json
import joblib
import logging
import argparse
from typing import Dict, Any, List, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from vision.civic_yolo import get_yolo_detector, VisionAnalysisResult
from llm.local_llm import get_llm_client
from pipeline_models import CivicPriorityPipeline, CivicDuplicateMLPipeline

logger = logging.getLogger("civicpulse.ml.predict")

SLA_MAP = {
    "CRITICAL": 6,
    "HIGH": 24,
    "MEDIUM": 72,
    "LOW": 168
}

VISION_DEFECT_MAP = {
    "pothole": ("Roads & Infrastructure", "Pothole", "HIGH", "Pothole on Road"),
    "damaged_road": ("Roads & Infrastructure", "Road Damage", "MEDIUM", "Damaged Road Surface"),
    "garbage": ("Waste Management", "Garbage Overflow", "MEDIUM", "Garbage Overflow on Street"),
    "overflowing_bin": ("Waste Management", "Garbage Overflow", "HIGH", "Overflowing Dustbin"),
    "water_leak": ("Water & Sewage", "Water Pipeline Leakage", "HIGH", "Water Pipeline Leakage"),
    "blocked_drain": ("Water & Sewage", "Drainage Overflow", "HIGH", "Drainage Blockage / Overflow"),
    "broken_streetlight": ("Electricity", "Streetlight Failure", "MEDIUM", "Broken Streetlight"),
    "fallen_tree": ("Parks", "Park Maintenance Issue", "HIGH", "Fallen Tree / Branch Obstruction"),
}


class CivicHybridPredictor:
    """
    Unified Production Predictor orchestrating Computer Vision, NLP, and XGBoost models.
    """

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = models_dir or os.path.join(CURRENT_DIR, "models")
        self._load_models()

    def _load_models(self):
        logger.info("Loading trained ML models from: %s", self.models_dir)
        self.dept_model = joblib.load(os.path.join(self.models_dir, "department_model.joblib"))
        self.cat_model = joblib.load(os.path.join(self.models_dir, "category_model.joblib"))
        self.prio_model: CivicPriorityPipeline = joblib.load(os.path.join(self.models_dir, "priority_model.joblib"))
        self.dup_model: CivicDuplicateMLPipeline = joblib.load(os.path.join(self.models_dir, "duplicate_model.joblib"))
        self.vision_model = get_yolo_detector()
        self.llm_client = get_llm_client()

    def predict_complaint(
        self,
        text: str = "",
        image_input: Optional[Any] = None,
        ward: Optional[str] = None,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        existing_complaints: Optional[List[dict]] = None
    ) -> Dict[str, Any]:
        """
        Executes full multi-modal inference across Vision, NLP, Priority, and Duplicate models.
        """
        raw_text = (text or "").strip()

        # 1. Computer Vision YOLOv8 Analysis
        vision_result: VisionAnalysisResult = self.vision_model.detect(image_input)

        # 2. Text NLP Predictions (TF-IDF + Linear SVM)
        text_for_nlp = raw_text if len(raw_text) > 0 else (vision_result.detected_problem if vision_result.has_visual_evidence else "civic issue")

        dept_probs_raw = self.dept_model.predict_proba([text_for_nlp])[0]
        dept_classes = list(self.dept_model.classes_)
        dept_probs = {cls_name: float(dept_probs_raw[i]) for i, cls_name in enumerate(dept_classes)}
        top_dept = max(dept_probs, key=dept_probs.get)
        dept_conf = round(float(dept_probs[top_dept]), 4)

        cat_probs_raw = self.cat_model.predict_proba([text_for_nlp])[0]
        cat_classes = list(self.cat_model.classes_)
        cat_probs = {cls_name: float(cat_probs_raw[i]) for i, cls_name in enumerate(cat_classes)}
        top_cat = max(cat_probs, key=cat_probs.get)
        cat_conf = round(float(cat_probs[top_cat]), 4)

        # 3. Vision Priority & Defect Fusion
        default_issue_title = f"{top_cat} Reported"
        if vision_result.has_visual_evidence and vision_result.detected_problem in VISION_DEFECT_MAP:
            v_dept, v_cat, v_prio, v_title = VISION_DEFECT_MAP[vision_result.detected_problem]
            # If text is sparse or vision confidence is strong, fuse vision defect
            if len(raw_text.split()) < 4 or dept_conf < 0.70:
                top_dept = v_dept
                top_cat = v_cat
                dept_conf = max(dept_conf, float(vision_result.confidence))
                cat_conf = max(cat_conf, float(vision_result.confidence))
                default_issue_title = v_title

        # Determine human-readable detected issue title (never blank)
        if len(raw_text) > 0:
            words = raw_text.split()
            if len(words) <= 6:
                detected_issue = raw_text
            else:
                detected_issue = f"{top_cat} - {' '.join(words[:4])}..."
        else:
            detected_issue = default_issue_title

        # 4. Priority Model (Random Forest)
        prio_probs_raw = self.prio_model.predict_proba([text_for_nlp], [top_dept], [top_cat])[0]
        prio_classes = list(self.prio_model.classes_)
        prio_probs = {cls_name: float(prio_probs_raw[i]) for i, cls_name in enumerate(prio_classes)}
        top_prio = max(prio_probs, key=prio_probs.get)
        prio_conf = round(float(prio_probs[top_prio]), 4)
        sla_hours = SLA_MAP.get(top_prio, 72)

        # Urgency score 0-100
        urgency_weights = {"LOW": 25.0, "MEDIUM": 55.0, "HIGH": 82.0, "CRITICAL": 98.0}
        urgency_score = round(sum(prio_probs.get(p, 0.0) * urgency_weights[p] for p in prio_classes), 1)

        # 5. Duplicate Detection (SBERT + XGBoost)
        duplicates = []
        if existing_complaints:
            query_obj = {
                "complaint": raw_text or detected_issue,
                "department": top_dept,
                "category": top_cat,
                "latitude": lat or 12.9784,
                "longitude": lng or 77.6408
            }
            duplicates = self.dup_model.find_duplicates(query_obj, existing_complaints, threshold=0.60)

        # 6. Local LLM / XAI Explanation Synthesis
        explanation = self.llm_client.generate_explanation(
            department=top_dept,
            category=top_cat,
            priority=top_prio,
            confidence=dept_conf,
            description=raw_text or detected_issue,
            vision_defect=vision_result.detected_problem if vision_result.has_visual_evidence else None,
            sla_hours=sla_hours,
            ward=ward
        )

        return {
            "detected_issue": detected_issue,
            "department": top_dept,
            "department_confidence": dept_conf,
            "category": top_cat,
            "category_confidence": cat_conf,
            "priority": top_prio,
            "priority_confidence": prio_conf,
            "sla_hours": sla_hours,
            "urgency_score": urgency_score,
            "explanation": explanation,
            "vision_analysis": vision_result.to_dict(),
            "duplicates_detected": duplicates,
            "provenance": {
                "department_algorithm": "TF-IDF + Linear SVM",
                "category_algorithm": "TF-IDF + Linear SVM",
                "priority_algorithm": "Random Forest Classifier",
                "duplicate_algorithm": "SBERT (Sentence-BERT) + XGBoost Classifier",
                "vision_algorithm": "Ultralytics YOLOv8 Defect Detector",
                "explanation_source": "Local LLM (Llama 3.2 / OpenRouter) + XAI"
            }
        }

    def predict_all(self, *args, **kwargs) -> Dict[str, Any]:
        return self.predict_complaint(*args, **kwargs)


# Global singleton instance
_predictor: Optional[CivicHybridPredictor] = None


def get_predictor() -> CivicHybridPredictor:
    global _predictor
    if _predictor is None:
        _predictor = CivicHybridPredictor()
    return _predictor


def cli():
    parser = argparse.ArgumentParser(description="CivicPulse Hybrid AI/ML Prediction Engine")
    parser.add_argument("text", nargs="?", help="Complaint text description")
    parser.add_argument("--image", help="Path to complaint photo or base64")
    parser.add_argument("--ward", help="Ward name (e.g. Indiranagar)")
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--lng", type=float, help="Longitude")
    parser.add_argument("--json", action="store_true", help="Output strict JSON")
    parser.add_argument("--interactive", action="store_true", help="Interactive terminal mode")

    args = parser.parse_args()
    predictor = get_predictor()

    if args.interactive:
        print("CivicPulse Interactive AI Prediction Mode (type 'exit' to quit)")
        while True:
            try:
                user_input = input("\nEnter Civic Complaint > ").strip()
                if not user_input or user_input.lower() in ["exit", "quit", "q"]:
                    break
                res = predictor.predict_complaint(user_input)
                print(f"[*] Issue:      {res['detected_issue']}")
                print(f"[*] Department: {res['department']} ({res['department_confidence']*100:.1f}%)")
                print(f"[*] Category:   {res['category']} ({res['category_confidence']*100:.1f}%)")
                print(f"[*] Priority:   {res['priority']} (SLA: {res['sla_hours']} Hours | Urgency: {res['urgency_score']}/100)")
                print(f"[*] Explanation: {res['explanation']}")
            except (KeyboardInterrupt, EOFError):
                break
        return

    if not args.text and not args.image:
        parser.print_help()
        sys.exit(1)

    result = predictor.predict_complaint(
        text=args.text or "",
        image_input=args.image,
        ward=args.ward,
        lat=args.lat,
        lng=args.lng
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 65)
        print("         CIVICPULSE HYBRID AI REPORT")
        print("=" * 65)
        print(f"Detected Issue: {result['detected_issue']}")
        print(f"Department:     {result['department']} (Confidence: {result['department_confidence']*100:.1f}%)")
        print(f"Category:       {result['category']} (Confidence: {result['category_confidence']*100:.1f}%)")
        print(f"Priority:       {result['priority']} (SLA: {result['sla_hours']} Hours | Urgency: {result['urgency_score']}/100)")
        if result['vision_analysis']['has_visual_evidence']:
            print(f"Vision YOLO:    {result['vision_analysis']['detected_problem']} ({result['vision_analysis']['confidence']*100:.1f}%)")
        print("-" * 65)
        print("Explanation:")
        print(f"{result['explanation']}")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    cli()
