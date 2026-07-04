from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from .activation import propagate, seed
from .assemble import select
from .consolidate import ConsolidationReport, run_consolidation
from .graph import MemoryGraph, Node
from .store import MemoryStore


class SecondBrain:
    """Public API surface for the secondbrain memory system."""

    def __init__(self, graph: Optional[MemoryGraph] = None) -> None:
        self.graph = graph or MemoryGraph()
        self.store = MemoryStore(graph=self.graph)

    def remember(self, text: str, metadata: Optional[dict] = None) -> str:
        """Store a new memory node. Returns the node_id."""
        node = self.store.remember(text, metadata=metadata)
        return node.id

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
        initial = seed(query_embedding, self.graph, top_k=8, floor=0.2)

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
        initial = seed(query_embedding, self.graph, top_k=8, floor=0.0)
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

    def consolidate(self) -> ConsolidationReport:
        """Run a full consolidation pass (sleep job)."""
        return run_consolidation(self.graph)


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


def _default() -> SecondBrain:
    if not hasattr(_default, "instance"):
        _default.instance = SecondBrain()
    return _default.instance
