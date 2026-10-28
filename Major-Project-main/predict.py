"""
CivicPulse Prediction CLI & Inference Interface
===============================================
Command-line and programmatic interface to run NLP classification and routing
inference on single, batch, or interactive civic complaint queries using `model.pkl`.
"""

import os
import sys
import json
import argparse
from typing import Optional, List, Dict, Any

from classifier import CivicComplaintClassifier, ClassificationResult


def format_terminal_output(result: ClassificationResult, query: str, ward: Optional[str] = None) -> str:
    """Formats classification result with clean terminal styling."""
    lines = []
    lines.append("\n" + "=" * 65)
    lines.append("  CIVICPULSE AI ISSUE CLASSIFICATION & ROUTING REPORT")
    lines.append("=" * 65)
    lines.append(f"Input Query:       \"{query}\"")
    if ward:
        lines.append(f"Ward / Location:   {ward}")
    lines.append("-" * 65)
    lines.append(f"Department:        {result.department} (Confidence: {result.department_confidence * 100:.1f}%)")
    lines.append(f"Priority:          {result.priority} (Confidence: {result.priority_confidence * 100:.1f}%)")
    lines.append(f"Urgency Score:     {result.urgency_score}/100 {'[CRITICAL ACTION]' if result.is_critical else ''}")
    lines.append(f"SLA Target Window: {result.sla_hours} Hours")
    lines.append(f"Detected Keywords: {', '.join(result.detected_keywords) if result.detected_keywords else 'None'}")
    lines.append(f"Severity Markers:  {', '.join(result.detected_severity_terms) if result.detected_severity_terms else 'Standard'}")
    lines.append("-" * 65)
    lines.append(f"AI Reasoning:\n{result.ai_reasoning}")
    lines.append("=" * 65 + "\n")
    return "\n".join(lines)


def predict_single(
    text: str,
    ward: Optional[str] = None,
    model_path: str = "model.pkl",
    output_json: bool = False
) -> Dict[str, Any]:
    """
    Predicts department, priority, and SLA for a single complaint string.
    """
    classifier = CivicComplaintClassifier(model_path)
    res = classifier.classify(text, ward=ward)
    if output_json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(format_terminal_output(res, text, ward))
    return res.to_dict()


def predict_batch(
    texts: List[str],
    model_path: str = "model.pkl",
    output_json: bool = False
) -> List[Dict[str, Any]]:
    """
    Predicts department, priority, and SLA for a list of complaint strings.
    """
    classifier = CivicComplaintClassifier(model_path)
    results = classifier.classify_batch(texts)
    dicts = [r.to_dict() for r in results]
    if output_json:
        print(json.dumps(dicts, indent=2))
    else:
        for idx, (t, r) in enumerate(zip(texts, results), 1):
            print(f"[{idx}/{len(texts)}] {format_terminal_output(r, t)}")
    return dicts


def run_interactive(model_path: str = "model.pkl"):
    """
    Launches an interactive shell loop to test predictions on real-time inputs.
    """
    classifier = CivicComplaintClassifier(model_path)
    print("\n" + "=" * 65)
    print("  CivicPulse Interactive Issue Classifier (Type 'exit' or 'q' to quit)")
    print("=" * 65)

    while True:
        try:
            query = input("\nEnter complaint text > ").strip()
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive mode.")
                break
            if not query:
                continue
            ward = input("Enter ward/location (optional, press Enter to skip) > ").strip() or None
            res = classifier.classify(query, ward=ward)
            print(format_terminal_output(res, query, ward))
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def main():
    parser = argparse.ArgumentParser(description="CivicPulse ML Prediction CLI")
    parser.add_argument("query", nargs="?", type=str, default=None, help="Complaint description text")
    parser.add_argument("--ward", "-w", type=str, default=None, help="Ward or location name")
    parser.add_argument("--model", "-m", type=str, default="model.pkl", help="Path to model.pkl file")
    parser.add_argument("--json", "-j", action="store_true", help="Output results in JSON format")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive prediction shell")
    parser.add_argument("--batch", "-b", nargs="+", help="Classify multiple complaint strings in batch")

    args = parser.parse_args()

    if args.interactive:
        run_interactive(args.model)
    elif args.batch:
        predict_batch(args.batch, model_path=args.model, output_json=args.json)
    elif args.query:
        predict_single(args.query, ward=args.ward, model_path=args.model, output_json=args.json)
    else:
        # Default demo prediction if no arguments passed
        default_query = "Massive water pipe leak flooding residential road in Indiranagar near metro station"
        print("[*] No query provided. Running demo prediction:")
        predict_single(default_query, ward="Indiranagar", model_path=args.model, output_json=args.json)


if __name__ == "__main__":
    main()
