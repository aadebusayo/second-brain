import numpy as np

from secondbrain.activation import base_level, propagate, seed
from secondbrain.graph import MemoryGraph


def test_spreading_activation_ranks_known_graph():
    graph = MemoryGraph()
    nodes = {}
    for node_id, text in [
        ("seed", "alpha"),
        ("neighbor", "alpha beta"),
        ("two_hop", "alpha beta gamma"),
        ("disconnected", "delta"),
    ]:
        node = graph.add_node(text=text)
        nodes[node_id] = node

    graph.add_edge(nodes["seed"].id, nodes["neighbor"].id, weight=0.95)
    graph.add_edge(nodes["neighbor"].id, nodes["two_hop"].id, weight=0.7)

    query_embedding = np.array([1.0, 0.0, 0.0])
    seed_activations = seed(query_embedding, graph, top_k=4, floor=0.0)
    scores = propagate(seed_activations, graph, gamma=0.7, hops=3)

    assert scores[nodes["neighbor"].id] > scores[nodes["two_hop"].id]
    assert scores[nodes["neighbor"].id] > scores[nodes["disconnected"].id]
    assert scores[nodes["seed"].id] > scores[nodes["disconnected"].id]


def test_base_level_rewards_recency():
    node = type("Node", (), {"access_log": [1.0, 2.0]})()
    assert base_level(node, decay=0.5) > 0.0
