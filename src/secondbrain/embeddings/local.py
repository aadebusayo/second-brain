from __future__ import annotations

import hashlib
import math
import re
from typing import List

from .base import BaseEmbeddingProvider


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """A deterministic local embedding provider with stronger lexical separation."""

    def embed(self, text: str) -> List[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return [0.0] * 16

        vector = [0.0] * 16
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            for index, byte in enumerate(digest):
                vector[index % len(vector)] += byte / 255.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return [0.0] * 16
        return [value / norm for value in vector]
