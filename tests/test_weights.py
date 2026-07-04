from secondbrain.graph import MemoryGraph
from secondbrain.weights import decay_all, reinforce


def test_reinforce_and_decay_change_edge_strength():
    graph = MemoryGraph()
    a = graph.add_node(text="a")
    b = graph.add_node(text="b")
    graph.add_edge(a.id, b.id, weight=0.2)

    # reinforce with co_activation=1.0 (full co-retrieval), eta=0.1
    reinforce(graph, a.id, b.id, co_activation=1.0, eta=0.1)
    assert graph.get_edge_weight(a.id, b.id) > 0.2

    # decay with aggressive lambda
    decay_all(graph, lam=2.0, dt=1.0)
    assert graph.get_edge_weight(a.id, b.id) < 0.2
