"""
CivicPulse Local LLM, OpenRouter & Explainability Interface
===========================================================
Generates natural language explanations, triage summaries, and dispatch
justifications using OpenRouter API, local Ollama (Llama 3.2), or a
zero-dependency transparent Explainable AI (XAI) fallback engine.
"""

import os
import json
import logging
import requests
from typing import Dict, Any, Optional

try:
    from llm.openrouter_client import get_openrouter_client
except Exception:
    try:
        from openrouter_client import get_openrouter_client
    except Exception:
        get_openrouter_client = None

logger = logging.getLogger("civicpulse.llm.local")


class LocalLLMClient:
    """
    Unified LLM Client supporting OpenRouter API, local Ollama,
    and fallback XAI narrative generators.
    """

    def __init__(
        self,
        ollama_url: Optional[str] = None,
        model_name: str = "llama3.2",
        openrouter_api_key: Optional[str] = None
    ):
        self.ollama_url = ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
        self.model_name = model_name or os.environ.get("LOCAL_LLM_MODEL", "llama3.2")
        self.openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()
        self.timeout_seconds = 4.0

    def generate_explanation(
        self,
        department: str,
        category: str,
        priority: str,
        confidence: float,
        description: str,
        vision_defect: Optional[str] = None,
        sla_hours: int = 72,
        ward: Optional[str] = None
    ) -> str:
        """
        Synthesizes a clear, human-readable explanation of why the issue was
        classified into the specified Department, Category, and Priority.
        Tiers:
          1. OpenRouter API (if configured)
          2. Local Ollama (if running)
          3. Deterministic XAI synthesizer
        """
        # 1. Try OpenRouter API first
        if get_openrouter_client:
            openrouter = get_openrouter_client()
            if openrouter.is_configured:
                or_exp = openrouter.generate_explanation(
                    department=department,
                    category=category,
                    priority=priority,
                    confidence=confidence,
                    description=description,
                    vision_defect=vision_defect,
                    sla_hours=sla_hours,
                    ward=ward
                )
                if or_exp:
                    return or_exp

        # 2. Try local Ollama if available
        ollama_result = self._query_ollama(
            department=department,
            category=category,
            priority=priority,
            confidence=confidence,
            description=description,
            vision_defect=vision_defect,
            sla_hours=sla_hours,
            ward=ward
        )
        if ollama_result:
            return ollama_result

        # 3. Fallback to local heuristic XAI synthesizer
        return self._generate_heuristic_explanation(
            department=department,
            category=category,
            priority=priority,
            confidence=confidence,
            description=description,
            vision_defect=vision_defect,
            sla_hours=sla_hours,
            ward=ward
        )

    def _query_ollama(
        self,
        department: str,
        category: str,
        priority: str,
        confidence: float,
        description: str,
        vision_defect: Optional[str],
        sla_hours: int,
        ward: Optional[str]
    ) -> Optional[str]:
        """Queries local Ollama HTTP endpoint."""
        prompt = (
            f"You are the AI triage explanation engine for a smart city civic platform. "
            f"Write a concise 1-2 sentence explanation justifying this complaint routing:\n"
            f"- Citizen Report: \"{description}\"\n"
            f"- Ward: {ward or 'Citywide'}\n"
            f"- ML Department: {department}\n"
            f"- Issue Category: {category}\n"
            f"- Priority: {priority}\n"
            f"- ML Confidence: {confidence * 100:.1f}%\n"
            f"- Visual Evidence: {vision_defect or 'None'}\n"
            f"- SLA Window: {sla_hours} Hours\n\n"
            f"Explain clearly why this department and priority were assigned. Output only the explanation."
        )

        try:
            url = f"{self.ollama_url.rstrip('/')}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 120
                }
            }
            resp = requests.post(url, json=payload, timeout=self.timeout_seconds)
            if resp.status_code == 200:
                data = resp.json()
                text = data.get("response", "").strip()
                if text:
                    return text
        except Exception as e:
            logger.debug("Local Ollama not reachable (%s), falling back.", e)
            return None

        return None

    def _generate_heuristic_explanation(
        self,
        department: str,
        category: str,
        priority: str,
        confidence: float,
        description: str,
        vision_defect: Optional[str],
        sla_hours: int,
        ward: Optional[str]
    ) -> str:
        """
        Deterministic, transparent XAI synthesizer.
        """
        conf_pct = int(confidence * 100) if confidence <= 1.0 else int(confidence)
        ward_phrase = f" in {ward}" if ward else ""
        vision_phrase = f" Cross-verified by computer vision detecting '{vision_defect}'." if vision_defect and vision_defect != "unknown" else ""

        # Priority impact rationale
        if priority == "CRITICAL":
            impact_reason = "presents an immediate hazard to public health and safety requiring emergency dispatch"
        elif priority == "HIGH":
            impact_reason = "causes severe public disruption and property degradation requiring expedited action"
        elif priority == "MEDIUM":
            impact_reason = "impacts standard civic amenities and requires scheduled field inspection"
        else:
            impact_reason = "represents a routine cosmetic/minor maintenance request"

        return (
            f"The complaint is classified under '{department}' ({category}) with {conf_pct}% confidence{ward_phrase}. "
            f"Assigned {priority} priority (SLA: {sla_hours}h) because the reported issue {impact_reason}.{vision_phrase}"
        )


# Global singleton instance
_llm_client: Optional[LocalLLMClient] = None


def get_llm_client() -> LocalLLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LocalLLMClient()
    return _llm_client


def explain_prediction(
    department: str,
    category: str,
    priority: str,
    confidence: float,
    description: str,
    vision_defect: Optional[str] = None,
    sla_hours: int = 72,
    ward: Optional[str] = None
) -> str:
    """Convenience helper to generate an explanation."""
    return get_llm_client().generate_explanation(
        department=department,
        category=category,
        priority=priority,
        confidence=confidence,
        description=description,
        vision_defect=vision_defect,
        sla_hours=sla_hours,
        ward=ward
    )
