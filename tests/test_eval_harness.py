from pathlib import Path

from secondbrain.eval import (
    benchmark_diverse_retrieval,
    benchmark_document_corpus,
    benchmark_memory_store,
    benchmark_retrieval_sizes,
    build_diverse_synthetic_graph,
    evaluate_retrieval,
    trace_activation_dynamics,
)


def test_eval_harness_reports_metrics():
    graph = build_diverse_synthetic_graph(24)
    report = evaluate_retrieval(graph, query="supply chain resilience", relevant_keywords=["supply", "logistics", "inventory"])
    assert report["precision@k"] >= 0.0
    assert report["recall@k"] >= 0.0
    assert report["mrr"] >= 0.0
    assert len(report["top_k_ids"]) == 5


def test_benchmark_retrieval_sizes_runs():
    results = benchmark_retrieval_sizes()
    assert len(results) == 3
    assert all(isinstance(item[1], dict) for item in results)


def test_memory_store_benchmark_reports_diverse_hits():
    report = benchmark_memory_store()
    assert report["precision@k"] >= 0.0
    assert report["recalled_relevant_items"] >= 0
    assert len(report["top_texts"]) == 5


def test_diverse_benchmark_spans_multiple_domains():
    report = benchmark_diverse_retrieval()
    assert report["scenario_count"] >= 20
    assert len({scenario["industry"] for scenario in report["scenarios"]}) >= 12


def test_document_corpus_benchmark_loads_real_documents_and_answers_queries():
    corpus_dir = Path(__file__).parent / "fixtures" / "knowledge"
    report = benchmark_document_corpus(corpus_dir)
    assert report["document_count"] >= 10
    assert report["scenario_count"] >= 5
    assert all(result["passed"] for result in report["results"])


def test_trace_activation_dynamics_produces_visualizable_output():
    graph = build_diverse_synthetic_graph(12)
    trace = trace_activation_dynamics(graph, query="supply chain resilience")
    assert trace["query"] == "supply chain resilience"
    assert "activation_trace" in trace
    assert "edge_trace" in trace
    assert len(trace["activation_trace"]) >= 1
    assert len(trace["edge_trace"]) >= 1
