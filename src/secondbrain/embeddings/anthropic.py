from __future__ import annotations

from typing import List

from .base import BaseEmbeddingProvider


class AnthropicEmbeddingProvider(BaseEmbeddingProvider):
    """Anthropic-backed embedding provider.

    This implementation intentionally raises an error unless the deployment
    provides a concrete adapter. The package should not silently claim support
    for a backend it cannot validate.
    """

    def embed(self, text: str) -> List[float]:
        raise RuntimeError(
            "Anthropic embeddings are not configured in this build. "
            "Install and configure a concrete provider adapter before use."
        )
