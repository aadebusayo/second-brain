from secondbrain.graph import MemoryGraph
from secondbrain.weights import decay_all, reinforce


def test_reinforce_and_decay_change_edge_strength():
    graph = MemoryGraph()
    a = graph.add_node(text="a")
    b = graph.add_node(text="b")
    graph.add_edge(a.id, b.id, weight=0.2)

    reinforce(graph, a.id, b.id, eta=0.5)
    assert graph.get_edge_weight(a.id, b.id) > 0.2

    decay_all(graph, lam=0.5, dt=2.0)
    assert graph.get_edge_weight(a.id, b.id) < 0.2
