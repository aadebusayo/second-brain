from __future__ import annotations

from typing import List

from .base import BaseEmbeddingProvider


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI-backed embedding provider.

    This implementation intentionally raises an error unless a concrete adapter
    is wired in by the deployment.
    """

    def embed(self, text: str) -> List[float]:
        raise RuntimeError(
            "OpenAI embeddings are not configured in this build. "
            "Install and configure a concrete provider adapter before use."
        )
