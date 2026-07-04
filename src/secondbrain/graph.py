from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import networkx as nx

from .embeddings.local import LocalEmbeddingProvider


@dataclass
class Node:
    id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    access_log: List[float] = field(default_factory=list)
    base_activation: float = 0.0


class MemoryGraph:
    def __init__(self) -> None:
        self._graph = nx.Graph()
        self._node_counter = 0

    def add_node(self, text: str, embedding: Optional[List[float]] = None, metadata: Optional[Dict[str, object]] = None) -> Node:
        self._node_counter += 1
        node_id = f"n{self._node_counter}"
        node_embedding = list(embedding) if embedding is not None else self._embed_text(text)
        node = Node(id=node_id, text=text, embedding=node_embedding, metadata=metadata or {}, access_log=[])
        self._graph.add_node(node_id, node=node)
        return node

    def add_edge(self, source_id: str, target_id: str, weight: float = 0.0) -> None:
        self._graph.add_edge(source_id, target_id, weight=float(weight))

    def remove_edge(self, source_id: str, target_id: str) -> None:
        if self._graph.has_edge(source_id, target_id):
            self._graph.remove_edge(source_id, target_id)

    def get_edge_weight(self, source_id: str, target_id: str) -> float:
        if not self._graph.has_edge(source_id, target_id):
            return 0.0
        return self._graph[source_id][target_id].get("weight", 0.0)

    def set_edge_weight(self, source_id: str, target_id: str, weight: float) -> None:
        if not self._graph.has_edge(source_id, target_id):
            self._graph.add_edge(source_id, target_id, weight=float(weight))
        else:
            self._graph[source_id][target_id]["weight"] = float(weight)

    def get_node(self, node_id: str) -> Optional[Node]:
        data = self._graph.nodes.get(node_id)
        return data.get("node") if data else None

    def list_nodes(self) -> List[Node]:
        return [
            data["node"]
            for _, data in self._graph.nodes(data=True)
            if "node" in data
        ]

    def neighbors(self, node_id: str) -> List[str]:
        if node_id not in self._graph:
            return []
        return list(self._graph.neighbors(node_id))

    def adjacency(self) -> Dict[str, List[Tuple[str, float]]]:
        return {
            node_id: [(nbr, data["weight"]) for nbr, data in self._graph[node_id].items()]
            for node_id in self._graph.nodes
        }

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    @staticmethod
    def _embed_text(text: str) -> List[float]:
        return LocalEmbeddingProvider().embed(text)
