from __future__ import annotations

from typing import List, Optional

from .activation import propagate, seed
from .graph import MemoryGraph, Node
from .store import MemoryStore


class SecondBrain:
    def __init__(self, graph: Optional[MemoryGraph] = None) -> None:
        self.graph = graph or MemoryGraph()
        self.store = MemoryStore(graph=self.graph)

    def remember(self, text: str, metadata: Optional[dict] = None) -> str:
        node = self.store.remember(text, metadata=metadata)
        return node.id

    def recall_naive(self, query: str, top_k: int = 3) -> List[Node]:
        return self.store.recall_naive(query, top_k=top_k)

    def recall(self, query: str, token_budget: int = 2000) -> List[Node]:
        return self.recall_naive(query, top_k=3)

    def explain(self, node_id: str) -> dict:
        node = self.graph.get_node(node_id)
        if node is None:
            return {"node_id": node_id, "reason": "missing"}
        query_embedding = node.embedding or [1.0, 0.0, 0.0]
        initial = seed(query_embedding, self.graph, top_k=4, floor=0.0)
        propagated = propagate(initial, self.graph, gamma=0.3, hops=2)
        return {
            "node_id": node_id,
            "seed_value": initial.get(node_id, 0.0),
            "final_score": propagated.get(node_id, 0.0),
            "trace": [{"node_id": node_id, "score": propagated.get(node_id, 0.0)}],
        }

    def consolidate(self) -> dict:
        return {"status": "ok", "nodes": self.graph.node_count()}


def remember(text: str, metadata: Optional[dict] = None) -> str:
    return _default().remember(text, metadata=metadata)


def recall(query: str, token_budget: int = 2000) -> List[Node]:
    return _default().recall(query, token_budget=token_budget)


def recall_naive(query: str, top_k: int = 3) -> List[Node]:
    return _default().recall_naive(query, top_k=top_k)


def explain(node_id: str) -> dict:
    return _default().explain(node_id)


def consolidate() -> dict:
    return _default().consolidate()


def _default() -> SecondBrain:
    if not hasattr(_default, "instance"):
        _default.instance = SecondBrain()
    return _default.instance
