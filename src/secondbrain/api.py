from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .activation import base_level, propagate, seed
from .assemble import select
from .consolidate import ConsolidationReport, run_consolidation
from .entity import Entity, EntityModel, ExtractionResult
from .graph import MemoryGraph, Node
from .llm import create_llm_client
from .store import MemoryStore
from .weights import reinforce


class SecondBrain:
    """Public API surface for the secondbrain memory system."""

    def __init__(self, graph: Optional[MemoryGraph] = None, llm_client: Any = None) -> None:
        self.graph = graph or MemoryGraph()
        self.store = MemoryStore(graph=self.graph)
        self.entities = EntityModel(graph=self.graph, embedding_provider=self.store.embedding_provider)
        self._llm_client = llm_client  # lazily created on first use
        self._nodes_since_consolidation = 0
        self._consolidate_every_n = int(os.environ.get("SECOND_BRAIN_CONSOLIDATE_EVERY_N", "100"))

    def remember(
        self,
        text: str,
        metadata: Optional[dict] = None,
        entities: Optional[List[str]] = None,
        auto_extract: bool = True,
        llm_client: Any = None,
    ) -> str:
        """
        Store a new memory node. Returns the node_id.

        If *entities* is provided, the node is linked to those entity
        nodes via 'mentions' edges.

        If *auto_extract* is True (default), entities and relations are
        extracted from the text using the configured LLM (DeepSeek by
        default) and automatically wired into the graph.
        """
        node = self.store.remember(text, metadata=metadata)

        if entities:
            for entity_id in entities:
                self.entities.link_to_node(entity_id, node.id, relation_type="mentions", weight=0.7)

        if auto_extract:
            client = llm_client or self._get_llm_client()
            if client is not None:
                try:
                    result = self.entities.extract_from_text(text, llm_client=client)
                    if result.entities:
                        self.entities.ingest_extraction(node.id, result)
                except Exception:
                    pass  # extraction failure is non-fatal

        # Volume-based auto-consolidation
        self._nodes_since_consolidation += 1
        if self._nodes_since_consolidation >= self._consolidate_every_n:
            self._nodes_since_consolidation = 0
            try:
                self.consolidate()
            except Exception:
                pass

        return node.id

    def _get_llm_client(self) -> Any:
        """Lazily create the default LLM client from Settings.

        Returns None if no LLM is configured or if the client was
        explicitly disabled (set to False by test helpers).
        """
        if self._llm_client is False:
            return None  # explicitly disabled
        if self._llm_client is not None:
            return self._llm_client  # already created or user-provided
        try:
            provider = self.store.settings.llm_provider
            api_key = self.store.settings.deepseek_api_key or None
            self._llm_client = create_llm_client(provider, api_key=api_key)
        except Exception:
            self._llm_client = False  # mark as unavailable
            return None
        return self._llm_client

    def recall_naive(self, query: str, top_k: int = 3) -> List[Node]:
        """Naive cosine-similarity recall (no activation spreading)."""
        return self.store.recall_naive(query, top_k=top_k)

    def recall(self, query: str, token_budget: int = 2000) -> List[Node]:
        """
        Full retrieval pipeline: embed query → seed → propagate →
        assemble under token budget. Returns the top matching nodes.
        """
        query_embedding = np.asarray(
            self.store.embedding_provider.embed(query), dtype=float
        )

        # Phase 1: seed with cosine similarity
        # Scale seed breadth with graph size so propagation has enough
        # candidates to work with — too few seeds starve the activation.
        graph_size = self.graph.node_count()
        seed_k = max(8, min(graph_size // 5, 100))
        initial = seed(query_embedding, self.graph, top_k=seed_k, floor=0.2)

        # Phase 2: spreading activation
        activated = propagate(initial, self.graph, gamma=0.6, hops=3)

        # Phase 3: assemble under token budget
        token_counts: Dict[str, int] = {}
        for node in self.graph.list_nodes():
            # Rough token estimate: ~1 token per 4 chars
            token_counts[node.id] = max(1, len(node.text) // 4)

        selected_ids = select(activated, token_counts, token_budget)

        return [
            node
            for node_id in selected_ids
            if (node := self.graph.get_node(node_id)) is not None
        ]

    def explain(self, node_id: str) -> dict:
        """
        Return the actual activation math that surfaced this node.

        Includes seed value, hop-by-hop contributions, base-level
        bonus, and final score — non-negotiable for debuggability.
        """
        node = self.graph.get_node(node_id)
        if node is None:
            return {"node_id": node_id, "reason": "missing"}

        query_embedding = (
            np.asarray(node.embedding, dtype=float)
            if node.embedding
            else np.zeros(16, dtype=float)
        )

        # Run the full pipeline and trace each hop
        seed_k = max(8, min(self.graph.node_count() // 5, 100))
        initial = seed(query_embedding, self.graph, top_k=seed_k, floor=0.0)
        seed_value = initial.get(node_id, 0.0)

        from .activation import base_level
        bl = base_level(node)

        # Re-run propagation one hop at a time to trace
        hop_trace: list[dict] = []
        current = {nid: float(s) for nid, s in initial.items()}
        for hop in range(3):
            current = propagate(current, self.graph, gamma=0.6, hops=1)
            hop_trace.append({
                "hop": hop,
                "score": round(current.get(node_id, 0.0), 6),
            })

        final_score = current.get(node_id, 0.0)

        # Neighbour contributions
        neighbour_contribs: dict[str, float] = {}
        for neighbor_id in self.graph.neighbors(node_id):
            weight = self.graph.get_edge_weight(node_id, neighbor_id)
            if weight > 0:
                neighbour_contribs[neighbor_id] = round(weight, 4)

        return {
            "node_id": node_id,
            "text": node.text,
            "seed_value": round(seed_value, 6),
            "base_level": round(bl, 6),
            "final_score": round(final_score, 6),
            "hop_trace": hop_trace,
            "neighbour_contributions": neighbour_contribs,
        }

    def consolidate(self, llm_client: Any = None) -> ConsolidationReport:
        """Run a full consolidation pass (sleep job)."""
        client = llm_client or self._get_llm_client()
        return run_consolidation(self.graph, llm_client=client)

    # ---- Entity model ----

    def add_entity(self, name: str, entity_type: str, properties: Optional[dict] = None) -> Entity:
        """Create a typed entity node in the graph (idempotent by name)."""
        return self.entities.add_entity(name, entity_type, properties=properties)

    def find_entity(self, name: str) -> Optional[Entity]:
        """Look up an entity by name (case-insensitive)."""
        return self.entities.find_entity(name)

    def list_entities(self, entity_type: Optional[str] = None) -> List[Entity]:
        """List all entities, optionally filtered by type."""
        return self.entities.list_entities(entity_type=entity_type)

    def add_relation(self, source_id: str, target_id: str, relation_type: str, weight: float = 0.8) -> None:
        """Add a typed relation between two entities."""
        self.entities.add_relation(source_id, target_id, relation_type, weight=weight)

    def get_entity_neighborhood(self, entity_id: str) -> dict:
        """Return linked entities, chunks, and stats for an entity."""
        return self.entities.get_entity_neighborhood(entity_id)

    def link_entity_to_node(self, entity_id: str, node_id: str, relation_type: str = "mentions") -> None:
        """Link an entity to a memory chunk node."""
        self.entities.link_to_node(entity_id, node_id, relation_type=relation_type)

    def extract_entities(self, node_id: str, llm_client: Any = None) -> ExtractionResult:
        """
        Extract entities and relations from a stored memory chunk.

        Use this for explicit, on-demand extraction on existing nodes.
        For automatic extraction at remember() time, use auto_extract=True.
        """
        node = self.graph.get_node(node_id)
        if node is None:
            return ExtractionResult()
        result = self.entities.extract_from_text(node.text, llm_client=llm_client)
        if result.entities:
            self.entities.ingest_extraction(node_id, result)
        return result

    # ---- Relevance-gated Hebbian learning ----

    def mark_relevant(self, node_id: str) -> None:
        """
        Mark a node as relevant (accessed now).

        Updates the access log so base_level activation rewards recency
        and frequency of confirmed-relevant retrievals.
        """
        node = self.graph.get_node(node_id)
        if node is not None:
            node.access_log.append(time.time())

    def reinforce_pair(self, a: str, b: str, co_activation: float = 1.0, eta: float = 0.1) -> None:
        """
        Strengthen the edge between two nodes.

        Call this ONLY when the agent/human has confirmed that both nodes
        were genuinely relevant to the same query — not as a blind
        side-effect of recall.

        This is the relevance gate that makes Hebbian learning actually
        improve retrieval quality instead of degrading it.
        """
        reinforce(self.graph, a, b, co_activation=co_activation, eta=eta)

    def reinforce_result_pairs(self, node_ids: List[str], eta: float = 0.05) -> int:
        """
        Reinforce edges between every pair in *node_ids*. Returns
        number of edges reinforced.

        Does NOT mark nodes as relevant (no access_log update) — this
        is a pure graph-learning operation. Use mark_relevant separately
        when the agent/human confirms retrieval relevance.
        """
        count = 0
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                self.reinforce_pair(node_ids[i], node_ids[j], co_activation=1.0, eta=eta)
                count += 1
        return count


# ---- module-level convenience functions (singleton) ----


def remember(text: str, metadata: Optional[dict] = None) -> str:
    return _default().remember(text, metadata=metadata)


def recall(query: str, token_budget: int = 2000) -> List[Node]:
    return _default().recall(query, token_budget=token_budget)


def recall_naive(query: str, top_k: int = 3) -> List[Node]:
    return _default().recall_naive(query, top_k=top_k)


def explain(node_id: str) -> dict:
    return _default().explain(node_id)


def consolidate() -> ConsolidationReport:
    return _default().consolidate()


def mark_relevant(node_id: str) -> None:
    return _default().mark_relevant(node_id)


def reinforce_pair(a: str, b: str, co_activation: float = 1.0, eta: float = 0.1) -> None:
    return _default().reinforce_pair(a, b, co_activation=co_activation, eta=eta)


def reinforce_result_pairs(node_ids: List[str], eta: float = 0.05) -> int:
    return _default().reinforce_result_pairs(node_ids, eta=eta)


def add_entity(name: str, entity_type: str, properties: Optional[dict] = None) -> Entity:
    return _default().add_entity(name, entity_type, properties=properties)


def find_entity(name: str) -> Optional[Entity]:
    return _default().find_entity(name)


def list_entities(entity_type: Optional[str] = None) -> List[Entity]:
    return _default().list_entities(entity_type=entity_type)


def add_relation(source_id: str, target_id: str, relation_type: str, weight: float = 0.8) -> None:
    return _default().add_relation(source_id, target_id, relation_type, weight=weight)


def get_entity_neighborhood(entity_id: str) -> dict:
    return _default().get_entity_neighborhood(entity_id)


def link_entity_to_node(entity_id: str, node_id: str, relation_type: str = "mentions") -> None:
    return _default().link_entity_to_node(entity_id, node_id, relation_type=relation_type)


def _default() -> SecondBrain:
    if not hasattr(_default, "instance"):
        _default.instance = SecondBrain()
    return _default.instance
