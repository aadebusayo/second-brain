from __future__ import annotations

import math
from typing import Any, List, Optional

from .base import BaseVectorStore


class InMemoryVectorStore(BaseVectorStore):
    """A simple in-memory vector implementation used for local development."""

    def __init__(self) -> None:
        self._entries: List[dict[str, Any]] = []

    def add(self, node_id: str, vector: List[float], payload: Optional[dict] = None) -> None:
        self._entries.append({"node_id": node_id, "vector": vector, "payload": payload or {}})

    def search(self, vector: List[float], top_k: int = 3) -> List[dict[str, Any]]:
        scored = []
        for entry in self._entries:
            score = self._cosine_similarity(vector, entry["vector"])
            scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
        if denom == 0:
            return 0.0
        numerator = sum(x * y for x, y in zip(a, b))
        return numerator / denom
