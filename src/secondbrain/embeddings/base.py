from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """Interface for embedding providers used by the memory package."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Convert a text value into an embedding vector."""
