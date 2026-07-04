from __future__ import annotations

import time
from typing import List, Optional
import numpy as np

from .activation import cosine_similarity
from .config import Settings
from .embeddings.anthropic import AnthropicEmbeddingProvider
from .embeddings.local import LocalEmbeddingProvider
from .embeddings.openai import OpenAIEmbeddingProvider
from .embeddings.sentence_transformers import SentenceTransformerProvider
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

        # Auto-wire: create edges to similar existing nodes via cosine similarity.
        # This is what makes the graph self-assemble without manual wiring —
        # each new node finds its natural neighbourhood on insertion.
        self._auto_wire(node)

        self._persist_node(node)
        self.logger.info("remembered node", extra={"trace": build_trace("remember", node_id=node.id, text_length=len(text))})
        return node

    def _auto_wire(self, node: Node) -> int:
        """Create edges from *node* to existing nodes whose embeddings are
        sufficiently similar. Returns the number of edges created."""
        if not node.embedding:
            return 0
        threshold = self.settings.wire_threshold
        new_embedding = np.asarray(node.embedding, dtype=float)
        wired = 0
        for existing in self._nodes:
            if existing.id == node.id or not existing.embedding:
                continue
            sim = cosine_similarity(new_embedding, np.asarray(existing.embedding, dtype=float))
            if sim >= threshold:
                self.graph.add_edge(node.id, existing.id, weight=float(sim))
                self._persist_edge(node.id, existing.id, float(sim))
                wired += 1
        if wired:
            self.logger.debug(
                "auto-wired edges",
                extra={"trace": build_trace("auto-wire", node_id=node.id, edges_created=wired)},
            )
        return wired

    def _persist_edge(self, source_id: str, target_id: str, weight: float, relation_type: str = "") -> None:
        """Persist an edge to the storage backend."""
        self.storage.save_edge(source_id, target_id, weight, relation_type=relation_type)

    def _auto_wire_to_previous(self, node: Node, candidates: List[Node]) -> int:
        """Wire *node* to a restricted set of *candidates* (used during bulk
        re-wiring on cold start). Returns the number of edges created."""
        if not node.embedding:
            return 0
        threshold = self.settings.wire_threshold
        new_embedding = np.asarray(node.embedding, dtype=float)
        wired = 0
        for candidate in candidates:
            if not candidate.embedding:
                continue
            sim = cosine_similarity(new_embedding, np.asarray(candidate.embedding, dtype=float))
            if sim >= threshold:
                self.graph.add_edge(node.id, candidate.id, weight=float(sim))
                wired += 1
        return wired

    def _build_embedding_provider(self):
        provider = self.settings.embedding_provider.lower()
        if provider == "anthropic":
            return AnthropicEmbeddingProvider()
        if provider == "openai":
            return OpenAIEmbeddingProvider()
        if provider == "sentence-transformers":
            return SentenceTransformerProvider()
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

        # Auto-wire loaded nodes to each other so the graph is connected
        # even after a cold start from persisted state.
        if len(self._nodes) > 1:
            wired = 0
            # First, try to load persisted edges
            persisted_edges = self.storage.load_edges()
            if persisted_edges:
                for edge in persisted_edges:
                    self.graph.add_edge(
                        edge["source_id"], edge["target_id"],
                        weight=edge["weight"],
                    )
                self.logger.info(
                    "loaded persisted edges",
                    extra={"trace": build_trace("load-edges", count=len(persisted_edges))},
                )
            else:
                # No persisted edges — auto-wire from scratch
                for i, node in enumerate(self._nodes):
                    wired += self._auto_wire_to_previous(node, self._nodes[:i])
                if wired:
                    self.logger.info(
                        "rewired persisted graph",
                        extra={"trace": build_trace("rewire", edges_created=wired)},
                    )

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


