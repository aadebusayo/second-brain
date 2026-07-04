import pytest

from secondbrain.activation import propagate, seed
from secondbrain.embeddings.anthropic import AnthropicEmbeddingProvider
from secondbrain.embeddings.local import LocalEmbeddingProvider
from secondbrain.embeddings.openai import OpenAIEmbeddingProvider
from secondbrain.graph import MemoryGraph
from secondbrain.store import MemoryStore
from secondbrain.vectors.lancedb import LanceDBVectorStore


def test_provider_errors_are_explicit_and_non_silent():
    with pytest.raises(RuntimeError):
        AnthropicEmbeddingProvider().embed("some text")
    with pytest.raises(RuntimeError):
        OpenAIEmbeddingProvider().embed("some text")


def test_lancedb_requires_real_connection():
    with pytest.raises(RuntimeError):
        LanceDBVectorStore()


def test_activation_handles_disconnected_graph_without_crashing():
    graph = MemoryGraph()
    for text in ["alpha", "beta", "gamma"]:
        graph.add_node(text=text)
    scores = propagate({}, graph, gamma=0.5, hops=2)
    assert isinstance(scores, dict)
    assert len(scores) == 3


def test_memory_store_can_handle_many_notes():
    store = MemoryStore()
    for i in range(50):
        store.remember(f"enterprise note {i}")
    results = store.recall_naive("enterprise note", top_k=5)
    assert len(results) == 5


def test_retrieval_is_stable_under_noise():
    graph = MemoryGraph()
    relevant = [graph.add_node(text="compliance policy") for _ in range(3)]
    noise = [graph.add_node(text="weather report") for _ in range(20)]
    for node in relevant:
        graph.add_edge(node.id, noise[0].id, weight=0.05)
    query_embedding = LocalEmbeddingProvider().embed("compliance policy")
    seed_activations = seed(query_embedding, graph, top_k=graph.node_count(), floor=0.0)
    scores = propagate(seed_activations, graph, gamma=0.4, hops=2)
    assert scores[relevant[0].id] >= scores[noise[0].id]
