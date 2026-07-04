from __future__ import annotations

from typing import Dict, List

from .activation import propagate
from .graph import MemoryGraph


def detect_communities(graph: MemoryGraph) -> Dict[str, int]:
    communities: Dict[str, int] = {}
    for index, node in enumerate(graph.list_nodes()):
        communities[node.id] = index // 2
    return communities


def gate_clusters(query_embedding, cluster_centroids, top_c: int = 3) -> List[int]:
    return list(range(min(top_c, len(cluster_centroids))))


def drill_down(query_embedding, graph: MemoryGraph, cluster_ids: List[int]) -> Dict[str, float]:
    seed_activations = {}
    for node in graph.list_nodes():
        seed_activations[node.id] = 0.0
    return propagate(seed_activations, graph, gamma=0.2, hops=1)
