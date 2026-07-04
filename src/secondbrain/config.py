from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Application settings for provider selection, persistence, observability,
    and the four algorithmic tuning knobs that matter for retrieval quality."""

    # Provider configuration
    embedding_provider: str = "local"
    vector_provider: str = "inmemory"
    storage_backend: str = "sqlite"
    db_path: str = "secondbrain.sqlite"
    log_level: str = "INFO"

    # Algorithmic tuning knobs (the four that actually need tuning)
    gamma: float = 0.6       # spreading activation damping factor
    eta: float = 0.1         # Hebbian reinforcement learning rate
    decay_rate: float = 0.01 # edge weight exponential decay (lambda)
    hops: int = 3            # number of activation propagation hops

    def __init__(self) -> None:
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local")
        vector_provider = os.getenv("VECTOR_PROVIDER", "inmemory")
        storage_backend = os.getenv("STORAGE_BACKEND", "sqlite")
        db_path = os.getenv("SECOND_BRAIN_DB_PATH", "secondbrain.sqlite")
        log_level = os.getenv("SECOND_BRAIN_LOG_LEVEL", "INFO")
        gamma = float(os.getenv("SECOND_BRAIN_GAMMA", "0.6"))
        eta = float(os.getenv("SECOND_BRAIN_ETA", "0.1"))
        decay_rate = float(os.getenv("SECOND_BRAIN_DECAY_RATE", "0.01"))
        hops = int(os.getenv("SECOND_BRAIN_HOPS", "3"))

        self._validate(embedding_provider, vector_provider, storage_backend, log_level,
                       gamma, eta, decay_rate, hops)

        object.__setattr__(self, "embedding_provider", embedding_provider)
        object.__setattr__(self, "vector_provider", vector_provider)
        object.__setattr__(self, "storage_backend", storage_backend)
        object.__setattr__(self, "db_path", db_path)
        object.__setattr__(self, "log_level", log_level)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "decay_rate", decay_rate)
        object.__setattr__(self, "hops", hops)

    @staticmethod
    def _validate(
        embedding_provider: str,
        vector_provider: str,
        storage_backend: str,
        log_level: str,
        gamma: float,
        eta: float,
        decay_rate: float,
        hops: int,
    ) -> None:
        allowed_embeddings = {"local", "anthropic", "openai"}
        allowed_vectors = {"inmemory", "lancedb"}
        allowed_storage = {"sqlite"}
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if embedding_provider not in allowed_embeddings:
            raise ValueError(f"unsupported embedding provider: {embedding_provider}")
        if vector_provider not in allowed_vectors:
            raise ValueError(f"unsupported vector provider: {vector_provider}")
        if storage_backend not in allowed_storage:
            raise ValueError(f"unsupported storage backend: {storage_backend}")
        if log_level.upper() not in allowed_levels:
            raise ValueError(f"unsupported log level: {log_level}")
        if gamma <= 0 or gamma > 1:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if eta <= 0 or eta > 1:
            raise ValueError(f"eta must be in (0, 1], got {eta}")
        if decay_rate < 0:
            raise ValueError(f"decay_rate must be non-negative, got {decay_rate}")
        if hops < 1:
            raise ValueError(f"hops must be >= 1, got {hops}")
