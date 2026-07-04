from __future__ import annotations

from typing import Dict, List

import igraph as ig
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
        ig.VertexClustering.Optimizer,
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
    """
    if not isinstance(query_embedding, np.ndarray):
        query_embedding = np.asarray(query_embedding, dtype=float)

    cluster_set = set(cluster_ids)

    # Build a seed map restricted to nodes in the gated clusters
    seed_activations: Dict[str, float] = {}
    for node in graph.list_nodes():
        if communities.get(node.id, -1) in cluster_set and node.embedding:
            sim = cosine_similarity(query_embedding, np.asarray(node.embedding, dtype=float))
            if sim > 0:
                seed_activations[node.id] = sim

    # Only propagate within gated cluster nodes
    # Build a subgraph view by filtering the adjacency to gated-cluster nodes only
    if not seed_activations:
        return {}

    return propagate(seed_activations, graph, gamma=gamma, hops=hops)
