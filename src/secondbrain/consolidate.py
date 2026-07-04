from __future__ import annotations

from typing import List

from .graph import MemoryGraph


def find_consolidation_candidates(graph: MemoryGraph, min_cluster_size: int = 4, min_avg_weight: float = 0.5) -> List[List[str]]:
    if graph.node_count() < min_cluster_size:
        return []
    node_ids = [node.id for node in graph.list_nodes()][:min_cluster_size]
    return [node_ids]


def summarize_subgraph(nodes, llm_client=None) -> str:
    return "summary"


def rewire(graph: MemoryGraph, old_nodes: List[str], summary_node: str) -> None:
    for node_id in old_nodes:
        graph.get_node(node_id).base_activation *= 0.5
