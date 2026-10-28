"""
CivicPulse OpenRouter API Integration Client
============================================
Connects to OpenRouter (https://openrouter.ai/api/v1) for state-of-the-art LLM
triage, multimodal civic photo analysis, structured routing, and reasoning generation.
Supports models like Llama 3.3, DeepSeek R1, Gemini 2.0 Flash, Claude 3.5, etc.
"""

import os
import json
import logging
import requests
from typing import Dict, Any, Optional, List

logger = logging.getLogger("civicpulse.llm.openrouter")

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

DEPARTMENTS = [
    "Water & Sewage",
    "Waste Management",
    "Traffic",
    "Police / Public Safety",
    "Electricity",
    "Roads & Infrastructure",
    "Parks"
]

PRIORITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class OpenRouterClient:
    """
    Client for OpenRouter AI API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()
        self.model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
        self.timeout = 15.0

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 5)

    def analyze_complaint(
        self,
        description: str,
        image_base64: Optional[str] = None,
        location_info: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Performs end-to-end multimodal complaint triage via OpenRouter.
        Returns structured dictionary conforming to CivicPulse schema.
        """
        if not self.is_configured:
            return None

        system_prompt = (
            "You are the senior AI Triage & Routing Intelligence engine for CivicPulse, "
            "a municipal Smart City platform. Analyze the citizen report and return STRICT JSON only, no markdown.\n"
            f"Valid Departments (choose exactly one): {DEPARTMENTS}.\n"
            f"Valid Priorities: {PRIORITIES}.\n"
            "Priority Guidelines:\n"
            "- CRITICAL: Immediate danger to human life or public safety (live electrical wire, structural collapse, toxic/sewage in drinking water, armed crime).\n"
            "- HIGH: Severe public infrastructure disruption (major water transmission pipe burst, deep dangerous road sinkhole/pothole, blocked major arterial traffic).\n"
            "- MEDIUM: Standard civic disruption (overflowing garbage bin, broken streetlight, uncleaned park bench, illegal parking).\n"
            "- LOW: Minor, routine cosmetic maintenance (faded paint, minor road marking issue, routine tree pruning).\n\n"
            "Output JSON schema with exact keys:\n"
            "{\n"
            '  "detected_issue": "Short descriptive summary",\n'
            '  "department": "One valid department",\n'
            '  "category": "Specific issue category label",\n'
            '  "priority": "LOW | MEDIUM | HIGH | CRITICAL",\n'
            '  "confidence": integer between 75 and 99,\n'
            '  "reasoning": "1-2 concise, factual sentences explaining the routing and priority justification",\n'
            '  "tags": ["tag1", "tag2", "tag3"],\n'
            '  "safety_flag": boolean true if hazard to human safety\n'
            "}"
        )

        user_content: List[Dict[str, Any]] = []
        text_prompt = f"Citizen Complaint Description: \"{description}\"\nLocation: {location_info or 'Not specified'}\nProvide triage JSON:"
        user_content.append({"type": "text", "text": text_prompt})

        # Add image if provided
        if image_base64:
            b64_clean = image_base64.strip()
            if "," in b64_clean and b64_clean.startswith("data:"):
                b64_clean = b64_clean.split(",", 1)[1]
            data_url = f"data:image/jpeg;base64,{b64_clean}"
            user_content.append({
                "type": "image_url",
                "image_url": {"url": data_url}
            })

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://civicpulse.org",
            "X-Title": "CivicPulse Smart City Platform"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.2,
            "max_tokens": 400
        }

        try:
            resp = requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                return self._parse_json(content)
            else:
                logger.warning(f"OpenRouter API error {resp.status_code}: {resp.text}")
                return None
        except Exception as e:
            logger.warning(f"OpenRouter connection failed: {e}")
            return None

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
    ) -> Optional[str]:
        """
        Generates an explainable AI narrative using OpenRouter.
        """
        if not self.is_configured:
            return None

        prompt = (
            f"You are the AI triage explanation engine for a smart city civic platform. "
            f"Write a concise 1-2 sentence explanation justifying this complaint routing:\n"
            f"- Citizen Report: \"{description}\"\n"
            f"- Ward: {ward or 'Citywide'}\n"
            f"- Department: {department}\n"
            f"- Issue Category: {category}\n"
            f"- Priority: {priority}\n"
            f"- Confidence: {confidence * 100:.1f}%\n"
            f"- Visual Evidence: {vision_defect or 'None'}\n"
            f"- SLA Window: {sla_hours} Hours\n\n"
            f"Explain clearly why this department and priority were assigned. Output only the plain text explanation."
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://civicpulse.org",
            "X-Title": "CivicPulse Smart City Platform"
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 150
        }

        try:
            resp = requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                return text
        except Exception as e:
            logger.warning(f"OpenRouter explanation failed: {e}")
            return None

        return None

    def _parse_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]
        try:
            return json.loads(cleaned)
        except Exception:
            return None


# Global singleton
_openrouter_client: Optional[OpenRouterClient] = None


def get_openrouter_client(api_key: Optional[str] = None, model: Optional[str] = None) -> OpenRouterClient:
    global _openrouter_client
    if _openrouter_client is None or api_key:
        _openrouter_client = OpenRouterClient(api_key, model)
    return _openrouter_client
