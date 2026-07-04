from __future__ import annotations

import math

from .graph import MemoryGraph


def reinforce(
    graph: MemoryGraph,
    i: str,
    j: str,
    co_activation: float = 1.0,
    eta: float = 0.1,
) -> None:
    """
    Hebbian reinforcement on co-retrieval.

    Formula (per spec):
        w(i,j) += eta * (co_activation - w(i,j))

    Called whenever nodes i and j are retrieved together in the same recall.
    """
    current = graph.get_edge_weight(i, j)
    new_weight = current + eta * (co_activation - current)
    graph.set_edge_weight(i, j, new_weight)
    graph.set_edge_weight(j, i, new_weight)


def decay_all(graph: MemoryGraph, lam: float = 0.01, dt: float = 1.0) -> None:
    """
    Exponential time-decay on all edges.

    Formula (per spec):
        w(i,j) *= exp(-lam * dt)

    Edges below a floor threshold (1e-6) are dropped.
    """
    edges_to_drop: list[tuple[str, str]] = []
    for source_id, neighbors in graph.adjacency().items():
        for neighbor_id, weight in neighbors:
            decayed = weight * math.exp(-lam * dt)
            if decayed < 1e-6:
                edges_to_drop.append((source_id, neighbor_id))
            else:
                graph.set_edge_weight(source_id, neighbor_id, decayed)
    for source_id, neighbor_id in edges_to_drop:
        graph.remove_edge(source_id, neighbor_id)
