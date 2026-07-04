from __future__ import annotations

import math
from typing import Any, List, Optional

from .base import BaseVectorStore


class LanceDBVectorStore(BaseVectorStore):
    """Adapter for LanceDB-backed vector storage.

    Accepts an optional LanceDB connection or table. Falls back gracefully
    to an in-memory list when no real connection is provided, so the package
    remains usable without LanceDB.
    """

    def __init__(self, connection: Optional[Any] = None, table: Optional[Any] = None) -> None:
        self.connection = connection
        self.table = table
        self._entries: List[dict[str, Any]] = []

    @property
    def _ready(self) -> bool:
        return self.connection is not None and self.table is not None

    def add(self, node_id: str, vector: List[float], payload: Optional[dict] = None) -> None:
        if self._ready and self.table is not None:
            try:
                self.table.add([{"node_id": node_id, "vector": vector, ** (payload or {})}])
                return
            except Exception:
                pass
        self._entries.append({"node_id": node_id, "vector": vector, "payload": payload or {}})

    def search(self, vector: List[float], top_k: int = 3) -> List[dict[str, Any]]:
        if self._ready and self.table is not None:
            try:
                results = self.table.search(vector).limit(top_k).to_list()
                return [
                    {"node_id": r["node_id"], "vector": r["vector"], "payload": r}
                    for r in results
                ]
            except Exception:
                pass
        return self._fallback_search(vector, top_k)

    def _fallback_search(self, vector: List[float], top_k: int) -> List[dict[str, Any]]:
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
        return sum(x * y for x, y in zip(a, b)) / denom
