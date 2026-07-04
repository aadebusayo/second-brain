from __future__ import annotations

import time
from typing import List, Optional
import numpy as np

from .activation import cosine_similarity
from .config import Settings
from .embeddings.anthropic import AnthropicEmbeddingProvider
from .embeddings.local import LocalEmbeddingProvider
from .embeddings.openai import OpenAIEmbeddingProvider
from .graph import MemoryGraph, Node
from .storage import StorageBackend
from .vectors.inmemory import InMemoryVectorStore
from .vectors.lancedb import LanceDBVectorStore
from .observability import build_trace, get_logger


class MemoryStore:
    """Durable-ish in-memory store with pluggable embedding and vector providers."""

    def __init__(self, graph: Optional[MemoryGraph] = None, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.graph = graph or MemoryGraph()
        self._nodes: List[Node] = []
        self.embedding_provider = self._build_embedding_provider()
        self.vector_store = self._build_vector_store()
        self.storage = StorageBackend(settings=self.settings)
        self.logger = get_logger("secondbrain.store")
        self._load_existing_nodes()

    def remember(self, text: str, metadata: Optional[dict] = None) -> Node:
        node = self.graph.add_node(text=text, metadata=metadata)
        node.access_log.append(time.time())
        node.base_activation = 1.0
        node.embedding = self.embedding_provider.embed(text)
        self.vector_store.add(node.id, node.embedding, {"text": text})
        self._nodes.append(node)
        self._persist_node(node)
        self.logger.info("remembered node", extra={"trace": build_trace("remember", node_id=node.id, text_length=len(text))})
        return node

    def _build_embedding_provider(self):
        provider = self.settings.embedding_provider.lower()
        if provider == "anthropic":
            return AnthropicEmbeddingProvider()
        if provider == "openai":
            return OpenAIEmbeddingProvider()
        return LocalEmbeddingProvider()

    def _build_vector_store(self):
        provider = self.settings.vector_provider.lower()
        if provider == "lancedb":
            return LanceDBVectorStore()
        return InMemoryVectorStore()

    def recall_naive(self, query: str, top_k: int = 3) -> List[Node]:
        started = time.time()
        query_vector = np.asarray(self.embedding_provider.embed(query), dtype=float)
        results: List[tuple[float, Node]] = []
        for node in self.graph.list_nodes():
            if not node.embedding:
                continue
            score = cosine_similarity(query_vector, np.asarray(node.embedding, dtype=float))
            results.append((score, node))
        results.sort(key=lambda item: item[0], reverse=True)
        ranked = [node for _, node in results[:top_k]]
        self.logger.info(
            "recall complete",
            extra={"trace": build_trace("recall", query_length=len(query), top_k=top_k, duration_seconds=round(time.time() - started, 6), node_count=len(ranked))},
        )
        return ranked

    def _load_existing_nodes(self) -> None:
        loaded = self.storage.load_nodes()
        for item in loaded:
            node = self.graph.add_node(text=item["text"], embedding=item.get("embedding"))
            node.id = item["node_id"]
            node.embedding = item.get("embedding") or self.embedding_provider.embed(item["text"])
            node.metadata = item.get("metadata", {})
            self.vector_store.add(node.id, node.embedding, {"text": item["text"]})
            self._nodes.append(node)

    def _persist_node(self, node: Node) -> None:
        self.storage.save_node(node.id, node.text, node.embedding or [], node.metadata)

    def status(self) -> dict:
        return {
            "backend": self.settings.storage_backend,
            "embedding_provider": self.settings.embedding_provider,
            "vector_provider": self.settings.vector_provider,
            "node_count": len(self._nodes),
            "graph_nodes": self.graph.node_count(),
        }


