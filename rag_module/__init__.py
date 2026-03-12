"""
RAG Module - Retrieval & Knowledge Grounding
GenAI Team 18 | Preksha's Module

Pipeline: Question → Embedding → Vector Search → Top-k Retrieved Documents
"""

from .dataset_loader import DatasetLoader
from .vector_db import VectorDB
from .retriever import Retriever

__all__ = ["DatasetLoader", "VectorDB", "Retriever"]
