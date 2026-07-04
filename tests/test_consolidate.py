from secondbrain.consolidate import find_consolidation_candidates
from secondbrain.graph import MemoryGraph


def test_find_consolidation_candidates_returns_dense_subgraph():
    graph = MemoryGraph()
    nodes = [graph.add_node(text=f"n{i}") for i in range(8)]

    # Create a dense clique of 5 nodes (fully interconnected)
    for i in range(5):
        for j in range(i + 1, 5):
            graph.add_edge(nodes[i].id, nodes[j].id, weight=0.9)

    # Attach 3 sparse nodes to one clique member only
    for i in range(5, 8):
        graph.add_edge(nodes[0].id, nodes[i].id, weight=0.3)

    candidates = find_consolidation_candidates(graph, min_cluster_size=4, min_avg_weight=0.7)
    assert candidates
    assert len(candidates[0]) >= 4
