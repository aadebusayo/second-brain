"""Sentence-transformers embedding provider for secondbrain.

Uses the all-MiniLM-L6-v2 model (384-dim) for much better semantic
discrimination than the 16-dim local hash-based provider.
"""

from __future__ import annotations

from typing import List

from secondbrain.embeddings.base import BaseEmbeddingProvider


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Embedding provider backed by sentence-transformers all-MiniLM-L6-v2."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, text: str) -> List[float]:
        return self._model.encode(text).tolist()
