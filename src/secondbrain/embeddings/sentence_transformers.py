"""Sentence-transformers embedding provider for secondbrain.

Uses all-MiniLM-L6-v2 (384-dim) for real semantic discrimination.
This replaces the 16-dim local hash-based provider as the default.
"""

from __future__ import annotations

from typing import List

from .base import BaseEmbeddingProvider


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """384-dim sentence-transformers embeddings (all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        return self._model.encode(text).tolist()
