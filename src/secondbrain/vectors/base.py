from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseVectorStore(ABC):
    """Interface for vector stores that can back similarity search."""

    @abstractmethod
    def add(self, node_id: str, vector: List[float], payload: Optional[dict] = None) -> None:
        """Add a vector entry for a node."""

    @abstractmethod
    def search(self, vector: List[float], top_k: int = 3) -> List[dict[str, Any]]:
        """Return the closest matching vectors for a query vector."""
