from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Auto-load .env from the project root or current working directory
_ENV_PATH = Path(".env")
if not _ENV_PATH.exists():
    _ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
if _ENV_PATH.exists():
    with open(_ENV_PATH, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                _key = _key.strip()
                _val = _val.strip().strip('"').strip("'")
                if _key not in os.environ:
                    os.environ[_key] = _val


@dataclass(frozen=True)
class Settings:
    """Application settings for provider selection, persistence, observability,
    and the four algorithmic tuning knobs that matter for retrieval quality."""

    # Provider configuration
    embedding_provider: str = "sentence-transformers"
    vector_provider: str = "inmemory"
    storage_backend: str = "sqlite"
    db_path: str = "secondbrain.sqlite"
    log_level: str = "INFO"

    # Algorithmic tuning knobs
    gamma: float = 0.6         # spreading activation damping factor
    eta: float = 0.1           # Hebbian reinforcement learning rate
    decay_rate: float = 0.01   # edge weight exponential decay (lambda)
    hops: int = 3              # number of activation propagation hops
    wire_threshold: float = 0.4  # min cosine sim to auto-wire an edge on remember()

    # LLM configuration
    llm_provider: str = "deepseek"
    deepseek_api_key: str = ""

    def __init__(self) -> None:
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
        vector_provider = os.getenv("VECTOR_PROVIDER", "inmemory")
        storage_backend = os.getenv("STORAGE_BACKEND", "sqlite")
        db_path = os.getenv("SECOND_BRAIN_DB_PATH", "secondbrain.sqlite")
        log_level = os.getenv("SECOND_BRAIN_LOG_LEVEL", "INFO")
        gamma = float(os.getenv("SECOND_BRAIN_GAMMA", "0.6"))
        eta = float(os.getenv("SECOND_BRAIN_ETA", "0.1"))
        decay_rate = float(os.getenv("SECOND_BRAIN_DECAY_RATE", "0.01"))
        hops = int(os.getenv("SECOND_BRAIN_HOPS", "3"))
        wire_threshold = float(os.getenv("SECOND_BRAIN_WIRE_THRESHOLD", "0.4"))
        llm_provider = os.getenv("SECOND_BRAIN_LLM_PROVIDER", "deepseek")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")

        self._validate(embedding_provider, vector_provider, storage_backend, log_level,
                       gamma, eta, decay_rate, hops, wire_threshold)

        object.__setattr__(self, "embedding_provider", embedding_provider)
        object.__setattr__(self, "vector_provider", vector_provider)
        object.__setattr__(self, "storage_backend", storage_backend)
        object.__setattr__(self, "db_path", db_path)
        object.__setattr__(self, "log_level", log_level)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "decay_rate", decay_rate)
        object.__setattr__(self, "hops", hops)
        object.__setattr__(self, "wire_threshold", wire_threshold)
        object.__setattr__(self, "llm_provider", llm_provider)
        object.__setattr__(self, "deepseek_api_key", deepseek_api_key)

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
        wire_threshold: float,
    ) -> None:
        allowed_embeddings = {"local", "anthropic", "openai", "sentence-transformers"}
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
        if wire_threshold < 0 or wire_threshold > 1:
            raise ValueError(f"wire_threshold must be in [0, 1], got {wire_threshold}")
