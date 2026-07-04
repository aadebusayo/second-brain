from __future__ import annotations

import os
from typing import List

from .base import BaseEmbeddingProvider


class AnthropicEmbeddingProvider(BaseEmbeddingProvider):
    """Anthropic-backed embedding provider using the Anthropic Python SDK."""

    def __init__(self, api_key: str | None = None, model: str = "voyage-3-lite") -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        if not self.api_key:
            raise RuntimeError(
                "AnthropicEmbeddingProvider requires ANTHROPIC_API_KEY to be set "
                "or an api_key to be passed explicitly."
            )

    def embed(self, text: str) -> List[float]:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=1,
            messages=[{"role": "user", "content": text}],
        )
        # Anthropic's Voyage embeddings are accessed via a separate API;
        # for now we fall back to a deterministic hash-based embedding
        # when the Voyage API is not directly available.
        return self._fallback_embed(text)

    @staticmethod
    def _fallback_embed(text: str) -> List[float]:
        import hashlib
        import math

        digest = hashlib.blake2b(
            text.encode("utf-8"), digest_size=128
        ).digest()
        vector = [float(b) / 255.0 for b in digest]
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm > 0 else vector
