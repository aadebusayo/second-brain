from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Application settings for provider selection, persistence, and observability."""

    embedding_provider: str = "local"
    vector_provider: str = "inmemory"
    storage_backend: str = "sqlite"
    db_path: str = "secondbrain.sqlite"
    log_level: str = "INFO"

    def __init__(self) -> None:
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local")
        vector_provider = os.getenv("VECTOR_PROVIDER", "inmemory")
        storage_backend = os.getenv("STORAGE_BACKEND", "sqlite")
        db_path = os.getenv("SECOND_BRAIN_DB_PATH", "secondbrain.sqlite")
        log_level = os.getenv("SECOND_BRAIN_LOG_LEVEL", "INFO")
        self._validate(embedding_provider, vector_provider, storage_backend, log_level)
        object.__setattr__(self, "embedding_provider", embedding_provider)
        object.__setattr__(self, "vector_provider", vector_provider)
        object.__setattr__(self, "storage_backend", storage_backend)
        object.__setattr__(self, "db_path", db_path)
        object.__setattr__(self, "log_level", log_level)

    @staticmethod
    def _validate(embedding_provider: str, vector_provider: str, storage_backend: str, log_level: str) -> None:
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
