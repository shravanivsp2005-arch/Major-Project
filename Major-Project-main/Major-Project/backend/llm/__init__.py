"""CivicPulse Local LLM Package"""
from llm.local_llm import LocalLLMClient, get_llm_client, explain_prediction

__all__ = [
    "LocalLLMClient",
    "get_llm_client",
    "explain_prediction"
]
