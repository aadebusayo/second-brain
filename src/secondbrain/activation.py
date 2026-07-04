from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .graph import MemoryGraph


def base_level(node, decay: float = 0.5) -> float:
    """
    ACT-R style base-level activation: ln(sum(t_k ** -decay)).
    Rewards both recency and frequency of access.
    """
    if not getattr(node, "access_log", None):
        return 0.0
    return float(sum((t ** -decay) for t in node.access_log))


def seed(query_embedding, graph: MemoryGraph, top_k: int = 8, floor: float = 0.2) -> Dict[str, float]:
    """Cosine-similarity seeding: thresholded top-k matches to the query embedding."""
    if not isinstance(query_embedding, np.ndarray):
        query_embedding = np.asarray(query_embedding, dtype=float)
    scores: Dict[str, float] = {}
    for node in graph.list_nodes():
        if not node.embedding:
            continue
        sim = cosine_similarity(query_embedding, np.asarray(node.embedding, dtype=float))
        if sim >= floor:
            scores[node.id] = sim
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k])


def propagate(
    seed_activations: Dict[str, float],
    graph: MemoryGraph,
    gamma: float = 0.6,
    hops: int = 3,
) -> Dict[str, float]:
    """
    Damped spreading activation.

    Formula (per spec):
        a_i(h+1) = a_i(0) + B(i) + gamma * sum(w_ij * a_j(h))

    Where:
        a_i(0) = seed activation (cosine similarity to query)
        B(i)   = base-level activation (recency/frequency bonus)
        gamma  = damping factor per hop
        w_ij   = edge weight from j to i

    Normalization is applied per hop to prevent runaway growth.
    """
    node_ids = [node.id for node in graph.list_nodes()]
    id_to_node = {node.id: node for node in graph.list_nodes()}

    # Initialise: every node starts with its seed value + base-level bonus
    activations: Dict[str, float] = {}
    for node_id in node_ids:
        a0 = float(seed_activations.get(node_id, 0.0))
        bl = base_level(id_to_node[node_id])
        activations[node_id] = a0 + bl

    # Keep seed scores around for the formula: a_i(0) is re-added each hop
    seed_scores = {node_id: float(score) for node_id, score in seed_activations.items()}

    for hop in range(hops):
        next_scores: Dict[str, float] = {}
        for node_id in node_ids:
            a0 = seed_scores.get(node_id, 0.0)
            bl = base_level(id_to_node[node_id])
            # Sum weighted contributions from all activated neighbors
            neighbour_contrib = 0.0
            for neighbor_id in graph.neighbors(node_id):
                weight = graph.get_edge_weight(node_id, neighbor_id)
                if weight <= 0:
                    continue
                neighbour_contrib += weight * activations.get(neighbor_id, 0.0)
            next_scores[node_id] = a0 + bl + gamma * neighbour_contrib

        # Per-hop normalisation to prevent runaway growth
        max_score = max(next_scores.values(), default=0.0)
        if max_score > 0:
            next_scores = {nid: s / max_score for nid, s in next_scores.items()}

        activations = {nid: float(s) for nid, s in next_scores.items()}

    return dict(sorted(activations.items(), key=lambda item: item[1], reverse=True))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    # Pad the shorter vector to match the longer one's dimension
    if a.size != b.size:
        max_dim = max(a.size, b.size)
        if a.size < max_dim:
            a = np.pad(a, (0, max_dim - a.size))
        if b.size < max_dim:
            b = np.pad(b, (0, max_dim - b.size))
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
