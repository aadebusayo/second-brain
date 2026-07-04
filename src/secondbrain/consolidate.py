from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .graph import MemoryGraph, Node


@dataclass
class ConsolidationReport:
    """Report produced after a consolidation pass."""

    candidates_found: int = 0
    summaries_created: int = 0
    edges_rewired: int = 0
    nodes_demoted: int = 0
    details: list[dict] = field(default_factory=list)


def find_consolidation_candidates(
    graph: MemoryGraph,
    min_cluster_size: int = 4,
    min_avg_weight: float = 0.5,
) -> List[List[str]]:
    """
    Find dense, frequently co-activated subgraphs that are candidates
    for consolidation into a summary node.

    Uses a greedy clustering approach: pick a high-degree seed node,
    expand to neighbours with above-threshold edge weights, stop when
    the average internal edge weight drops below min_avg_weight.
    """
    if graph.node_count() < min_cluster_size:
        return []

    nodes_by_degree = sorted(
        graph.list_nodes(),
        key=lambda n: len(graph.neighbors(n.id)),
        reverse=True,
    )

    visited: set[str] = set()
    candidates: List[List[str]] = []

    for seed in nodes_by_degree:
        if seed.id in visited:
            continue

        cluster: list[str] = [seed.id]
        visited.add(seed.id)

        # Expand to strongly-connected neighbours
        for neighbor_id in graph.neighbors(seed.id):
            if neighbor_id in visited:
                continue
            weight = graph.get_edge_weight(seed.id, neighbor_id)
            if weight >= min_avg_weight:
                cluster.append(neighbor_id)
                visited.add(neighbor_id)

        if len(cluster) >= min_cluster_size:
            candidates.append(cluster)

    return candidates


def summarize_subgraph(
    nodes: List[Node],
    llm_client=None,
) -> Node:
    """
    Summarize a dense subgraph into a single compressed summary node.

    If an LLM client is provided, uses it to generate the summary text.
    Otherwise falls back to concatenation of node texts.
    """
    texts = [node.text for node in nodes if node.text]
    if llm_client is not None:
        try:
            prompt = (
                "Summarise the following related memory fragments into a single, "
                "concise paragraph that preserves all key information:\n\n"
                + "\n---\n".join(texts)
            )
            response = llm_client.messages.create(
                model=getattr(llm_client, "model", "claude-sonnet-5"),
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            summary_text = response.content[0].text
        except Exception:
            summary_text = " | ".join(texts)
    else:
        summary_text = " | ".join(texts)

    # Inherit an average embedding from the cluster
    import numpy as np

    embeddings = [
        np.asarray(node.embedding)
        for node in nodes
        if node.embedding and len(node.embedding) > 0
    ]
    if embeddings:
        avg_embedding = np.mean(np.stack(embeddings), axis=0).tolist()
    else:
        avg_embedding = None

    return Node(
        id="",
        text=summary_text,
        embedding=avg_embedding,
        metadata={"consolidated_from": [n.id for n in nodes]},
    )


def rewire(
    graph: MemoryGraph,
    old_nodes: List[str],
    summary_node_id: str,
) -> None:
    """
    Point external edges at the summary node and demote originals.

    Demotes (doesn't delete) the originals — keeps provenance while
    lowering their base activation so they won't surface in retrieval
    unless specifically relevant.
    """
    external_edges: dict[str, float] = {}

    for old_id in old_nodes:
        node = graph.get_node(old_id)
        if node is None:
            continue
        # Demote: halve base activation
        node.base_activation *= 0.5

        # Collect external edges (to nodes not in the old set)
        for neighbor_id in graph.neighbors(old_id):
            if neighbor_id not in old_nodes:
                weight = graph.get_edge_weight(old_id, neighbor_id)
                current = external_edges.get(neighbor_id, 0.0)
                external_edges[neighbor_id] = max(current, weight)

    # Wire the summary node to external neighbours
    for neighbor_id, weight in external_edges.items():
        graph.add_edge(summary_node_id, neighbor_id, weight=weight)


def run_consolidation(
    graph: MemoryGraph,
    llm_client=None,
    min_cluster_size: int = 4,
    min_avg_weight: float = 0.5,
) -> ConsolidationReport:
    """
    Run a full consolidation pass: find candidates, summarise each,
    rewire edges, demote originals.
    """
    report = ConsolidationReport()

    candidates = find_consolidation_candidates(
        graph, min_cluster_size=min_cluster_size, min_avg_weight=min_avg_weight
    )
    report.candidates_found = len(candidates)

    for cluster_ids in candidates:
        nodes = [graph.get_node(nid) for nid in cluster_ids]
        nodes = [n for n in nodes if n is not None]
        if len(nodes) < min_cluster_size:
            continue

        summary_node = summarize_subgraph(nodes, llm_client=llm_client)
        summary_node = graph.add_node(
            text=summary_node.text,
            embedding=summary_node.embedding,
            metadata=summary_node.metadata,
        )
        report.summaries_created += 1

        rewire(graph, cluster_ids, summary_node.id)
        report.nodes_demoted += len(cluster_ids)
        report.edges_rewired += len([
            nbr for nid in cluster_ids
            for nbr in graph.neighbors(nid)
            if nbr not in cluster_ids
        ])

        report.details.append({
            "old_nodes": cluster_ids,
            "summary_node": summary_node.id,
            "summary_text": summary_node.text[:200],
        })

    return report
