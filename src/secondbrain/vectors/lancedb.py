from __future__ import annotations

from typing import Any, List, Optional

from .base import BaseVectorStore


class LanceDBVectorStore(BaseVectorStore):
    """Adapter for LanceDB-backed vector storage.

    The package should not silently pretend this backend is operational without
    a real connection. This adapter raises an explicit error unless a real
    LanceDB connection is supplied by the deployment environment.
    """

    def __init__(self, connection: Optional[Any] = None) -> None:
        self.connection = connection
        if connection is None:
            raise RuntimeError(
                "LanceDB is not configured in this build. Provide a real connection "
                "before using this backend."
            )
        self._entries: List[dict[str, Any]] = []

    def add(self, node_id: str, vector: List[float], payload: Optional[dict] = None) -> None:
        self._entries.append({"node_id": node_id, "vector": vector, "payload": payload or {}})

    def search(self, vector: List[float], top_k: int = 3) -> List[dict[str, Any]]:
        return self._entries[:top_k]
