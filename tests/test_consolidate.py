from secondbrain.consolidate import find_consolidation_candidates
from secondbrain.graph import MemoryGraph


def test_find_consolidation_candidates_returns_dense_subgraph():
    graph = MemoryGraph()
    nodes = [graph.add_node(text=f"n{i}") for i in range(6)]
    for i in range(5):
        graph.add_edge(nodes[i].id, nodes[i + 1].id, weight=0.9)

    candidates = find_consolidation_candidates(graph, min_cluster_size=4, min_avg_weight=0.7)
    assert candidates
    assert len(candidates[0]) >= 4
