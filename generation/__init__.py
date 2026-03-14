"""
Generation Module — LLM-based Response Generation + Claim Extraction
Navya's Module | GenAI Team 18

Pipeline: Question + Retrieved Docs → LLM → Answer → Atomic Claims
"""

from .rag_generator import RAGGenerator, LLMBackend, OpenAIBackend, HuggingFaceBackend
from .claim_extractor import ClaimExtractor, ClaimExtractorAdvanced

__all__ = [
    "RAGGenerator",
    "LLMBackend",
    "OpenAIBackend",
    "HuggingFaceBackend",
    "ClaimExtractor",
    "ClaimExtractorAdvanced",
]
