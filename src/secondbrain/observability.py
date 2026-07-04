from __future__ import annotations

import logging
from typing import Any, Dict


def get_logger(name: str) -> logging.Logger:
    """Return a structured application logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def build_trace(event: str, **details: Any) -> Dict[str, Any]:
    """Create a structured trace payload for retrieval and persistence events."""
    return {"event": event, **details}
