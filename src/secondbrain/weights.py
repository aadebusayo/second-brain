from __future__ import annotations

import math

from .graph import MemoryGraph


def reinforce(graph: MemoryGraph, i: str, j: str, eta: float = 0.1) -> None:
    current = graph.get_edge_weight(i, j)
    new_weight = current + eta * (0.5 - current)
    graph.set_edge_weight(i, j, new_weight)
    graph.set_edge_weight(j, i, new_weight)


def decay_all(graph: MemoryGraph, lam: float = 0.01, dt: float = 1.0) -> None:
    for source_id, neighbors in graph.adjacency().items():
        for neighbor_id, weight in neighbors:
            decayed = weight * math.exp(-lam * dt)
            if decayed < 1e-6:
                graph._graph.remove_edge(source_id, neighbor_id)
            else:
                graph.set_edge_weight(source_id, neighbor_id, decayed)
