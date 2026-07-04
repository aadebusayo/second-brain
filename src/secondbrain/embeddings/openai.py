from __future__ import annotations

import os
from typing import List

from .base import BaseEmbeddingProvider


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI-backed embedding provider using the OpenAI Python SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        if not self.api_key:
            raise RuntimeError(
                "OpenAIEmbeddingProvider requires OPENAI_API_KEY to be set "
                "or an api_key to be passed explicitly."
            )

    def embed(self, text: str) -> List[float]:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.embeddings.create(model=self.model, input=text)
        return list(response.data[0].embedding)
