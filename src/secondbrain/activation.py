from __future__ import annotations

from typing import Dict

import numpy as np

from .graph import MemoryGraph


def base_level(node, decay: float = 0.5) -> float:
    if not getattr(node, "access_log", None):
        return 0.0
    return float(sum((t ** -decay) for t in node.access_log))


def seed(query_embedding, graph: MemoryGraph, top_k: int = 8, floor: float = 0.2) -> Dict[str, float]:
    if not isinstance(query_embedding, np.ndarray):
        query_embedding = np.asarray(query_embedding, dtype=float)
    query_embedding = _normalize_vector(query_embedding)
    scores: Dict[str, float] = {}
    for node in graph.list_nodes():
        if not node.embedding:
            continue
        sim = cosine_similarity(query_embedding, np.asarray(node.embedding, dtype=float))
        if sim >= floor:
            scores[node.id] = sim
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k])


def propagate(seed_activations: Dict[str, float], graph: MemoryGraph, gamma: float = 0.6, hops: int = 3) -> Dict[str, float]:
    activations = {node_id: float(score) for node_id, score in seed_activations.items()}
    node_ids = [node.id for node in graph.list_nodes()]
    seed_scores = {node_id: float(score) for node_id, score in seed_activations.items()}
    for _ in range(hops):
        next_scores = {node_id: activations.get(node_id, 0.0) for node_id in node_ids}
        for node_id, activation in list(activations.items()):
            for neighbor_id in graph.neighbors(node_id):
                weight = graph.get_edge_weight(node_id, neighbor_id)
                if weight <= 0:
                    continue
                next_scores[neighbor_id] = next_scores.get(neighbor_id, 0.0) + gamma * activation * weight
        for node_id, seed_score in seed_scores.items():
            next_scores[node_id] = max(next_scores.get(node_id, 0.0), seed_score)
        max_score = max(next_scores.values(), default=0.0)
        if max_score > 0:
            next_scores = {node_id: score / max_score for node_id, score in next_scores.items()}
        activations = {node_id: float(score) for node_id, score in next_scores.items()}
    return dict(sorted(activations.items(), key=lambda item: item[1], reverse=True))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    a = _normalize_vector(np.asarray(a, dtype=float))
    b = _normalize_vector(np.asarray(b, dtype=float))
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    if vector.size == 0:
        return vector
    if vector.size != 16:
        padded = np.zeros(16, dtype=float)
        padded[: min(vector.size, 16)] = vector[: min(vector.size, 16)]
        vector = padded
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm
