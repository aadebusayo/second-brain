from __future__ import annotations

from typing import Dict, List

import igraph as ig
import leidenalg
import numpy as np
from leidenalg import find_partition

from .activation import cosine_similarity, propagate
from .graph import MemoryGraph


def detect_communities(graph: MemoryGraph) -> Dict[str, int]:
    """
    Run Leiden community detection on the graph.

    Uses python-igraph + leidenalg to find densely connected communities.
    Returns a mapping of node_id -> cluster_id.
    """
    node_ids = [node.id for node in graph.list_nodes()]
    if len(node_ids) < 3:
        return {node_id: 0 for node_id in node_ids}

    # Build igraph from networkx graph
    ig_graph = ig.Graph()
    ig_graph.add_vertices(len(node_ids))
    id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

    edges: list[tuple[int, int]] = []
    edge_weights: list[float] = []
    for source_id, neighbors in graph.adjacency().items():
        for neighbor_id, weight in neighbors:
            if id_to_idx[source_id] < id_to_idx[neighbor_id]:
                edges.append((id_to_idx[source_id], id_to_idx[neighbor_id]))
                edge_weights.append(max(weight, 0.0))

    if not edges:
        return {node_id: idx // 2 for idx, node_id in enumerate(node_ids)}

    ig_graph.add_edges(edges)

    # Leiden with ModularityVertexPartition
    partition = find_partition(
        ig_graph,
        leidenalg.ModularityVertexPartition,
        weights=edge_weights if edge_weights else None,
        n_iterations=2,
    )

    return {node_id: int(partition.membership[id_to_idx[node_id]]) for node_id in node_ids}


def cluster_centroids(
    graph: MemoryGraph,
    communities: Dict[str, int],
) -> Dict[int, np.ndarray]:
    """
    Compute centroids for each cluster as the mean embedding of its members.
    """
    cluster_vectors: Dict[int, list[np.ndarray]] = {}
    for node in graph.list_nodes():
        if node.embedding:
            cid = communities.get(node.id, 0)
            cluster_vectors.setdefault(cid, []).append(np.asarray(node.embedding, dtype=float))

    centroids: Dict[int, np.ndarray] = {}
    for cid, vectors in cluster_vectors.items():
        if vectors:
            centroids[cid] = np.mean(np.stack(vectors), axis=0)
    return centroids


def gate_clusters(
    query_embedding,
    cluster_centroids: Dict[int, np.ndarray],
    top_c: int = 3,
) -> List[int]:
    """
    Cheap first pass: rank clusters by cosine similarity of query to centroid.
    Returns the top_c most relevant cluster IDs.
    """
    if not isinstance(query_embedding, np.ndarray):
        query_embedding = np.asarray(query_embedding, dtype=float)

    scored: list[tuple[float, int]] = []
    for cid, centroid in cluster_centroids.items():
        sim = cosine_similarity(query_embedding, centroid)
        scored.append((sim, cid))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [cid for _, cid in scored[:top_c]]


def drill_down(
    query_embedding,
    graph: MemoryGraph,
    cluster_ids: List[int],
    communities: Dict[str, int],
    gamma: float = 0.6,
    hops: int = 3,
) -> Dict[str, float]:
    """
    Run full spreading activation ONLY inside the gated clusters.
    This is what keeps retrieval cheap as the graph grows.

    Builds a filtered subgraph adjacency and runs damped propagation
    restricted to nodes whose community is in *cluster_ids*.
    """
    if not isinstance(query_embedding, np.ndarray):
        query_embedding = np.asarray(query_embedding, dtype=float)

    cluster_set = set(cluster_ids)

    # Collect nodes in the gated clusters
    gated_nodes = [
        n for n in graph.list_nodes()
        if communities.get(n.id, -1) in cluster_set and n.embedding
    ]
    gated_ids = {n.id for n in gated_nodes}

    if not gated_nodes:
        return {}

    # Seed: cosine similarity for gated nodes only
    seed_activations: Dict[str, float] = {}
    for node in gated_nodes:
        sim = cosine_similarity(query_embedding, np.asarray(node.embedding, dtype=float))
        if sim > 0:
            seed_activations[node.id] = sim

    if not seed_activations:
        return {}

    # Propagate only within gated cluster nodes (filtered adjacency)
    node_ids = [n.id for n in gated_nodes]
    id_to_node = {n.id: n for n in gated_nodes}

    from .activation import base_level

    activations: Dict[str, float] = {}
    for node_id in node_ids:
        a0 = float(seed_activations.get(node_id, 0.0))
        bl = base_level(id_to_node[node_id])
        activations[node_id] = a0 + bl

    seed_scores = {nid: float(s) for nid, s in seed_activations.items()}

    for _hop in range(hops):
        next_scores: Dict[str, float] = {}
        for node_id in node_ids:
            a0 = seed_scores.get(node_id, 0.0)
            bl = base_level(id_to_node[node_id])
            neighbour_contrib = 0.0
            for neighbor_id in graph.neighbors(node_id):
                if neighbor_id not in gated_ids:
                    continue  # skip nodes outside gated clusters
                weight = graph.get_edge_weight(node_id, neighbor_id)
                if weight <= 0:
                    continue
                neighbour_contrib += weight * activations.get(neighbor_id, 0.0)
            next_scores[node_id] = a0 + bl + gamma * neighbour_contrib

        max_score = max(next_scores.values(), default=0.0)
        if max_score > 0:
            next_scores = {nid: s / max_score for nid, s in next_scores.items()}
        activations = {nid: float(s) for nid, s in next_scores.items()}

    return dict(sorted(activations.items(), key=lambda item: item[1], reverse=True))
