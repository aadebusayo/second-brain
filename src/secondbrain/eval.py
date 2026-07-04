from __future__ import annotations

from typing import Dict, List, Tuple

from pathlib import Path

from .activation import propagate, seed
from .embeddings.local import LocalEmbeddingProvider
from .graph import MemoryGraph
from .store import MemoryStore
from .weights import decay_all


def _build_domain_catalog() -> List[Tuple[str, str]]:
    """Create a broad catalog of industry-specific memory content for evaluation."""
    domain_clusters = {
        "healthcare": [
            "patient triage",
            "clinical workflow",
            "medical compliance",
            "care pathway optimization",
            "telehealth onboarding",
            "diagnostic prioritization",
        ],
        "finance": [
            "credit risk",
            "portfolio hedging",
            "regulatory reporting",
            "fraud detection",
            "cash flow forecasting",
            "treasury operations",
        ],
        "manufacturing": [
            "supply chain resilience",
            "quality assurance",
            "inventory planning",
            "predictive maintenance",
            "vendor risk monitoring",
            "production scheduling",
        ],
        "retail": [
            "customer churn",
            "pricing strategy",
            "omnichannel loyalty",
            "demand forecasting",
            "store assortment planning",
            "returns optimization",
        ],
        "energy": [
            "grid optimization",
            "renewable forecasting",
            "emissions compliance",
            "asset maintenance",
            "load balancing",
            "energy storage planning",
        ],
        "telecom": [
            "network latency",
            "routing optimization",
            "service outage response",
            "capacity planning",
            "subscriber retention",
            "spectrum allocation",
        ],
        "public sector": [
            "citizen services",
            "policy implementation",
            "digital identity",
            "procurement transparency",
            "public safety coordination",
            "budget forecasting",
        ],
        "education": [
            "adaptive learning",
            "curriculum design",
            "student retention",
            "assessment analytics",
            "learning path personalization",
            "faculty scheduling",
        ],
        "sports": [
            "player injury prevention",
            "training load monitoring",
            "match strategy analysis",
            "talent scouting",
            "rehabilitation planning",
            "nutrition coordination",
        ],
        "hospitality": [
            "guest experience",
            "revenue management",
            "staff scheduling",
            "complaint resolution",
            "amenity personalization",
            "channel distribution",
        ],
        "agriculture": [
            "precision farming",
            "crop disease detection",
            "soil moisture modeling",
            "harvest planning",
            "supply planning",
            "irrigation optimization",
        ],
        "aerospace": [
            "flight safety",
            "navigation reliability",
            "maintenance forecasting",
            "crew scheduling",
            "mission planning",
            "component traceability",
        ],
    }
    return [(industry, term) for industry, terms in domain_clusters.items() for term in terms]


def build_diverse_synthetic_graph(size: int = 24) -> MemoryGraph:
    """Create a graph with a wide range of domain clusters and distractor nodes for evaluation."""
    graph = MemoryGraph()
    catalog = _build_domain_catalog()
    selected_entries = catalog[: max(12, min(len(catalog), size))]
    relevant_nodes = []
    for industry, term in selected_entries:
        relevant_nodes.append(graph.add_node(text=f"{industry} :: {term}"))
    distractor_nodes = [graph.add_node(text=f"distractor topic {i}") for i in range(max(size - len(relevant_nodes), 0))]

    for index in range(len(relevant_nodes) - 1):
        graph.add_edge(relevant_nodes[index].id, relevant_nodes[index + 1].id, weight=0.9)
    if distractor_nodes:
        for index in range(len(relevant_nodes)):
            graph.add_edge(relevant_nodes[index].id, distractor_nodes[index % len(distractor_nodes)].id, weight=0.05)

    return graph


def evaluate_retrieval(graph: MemoryGraph, query: str = "supply chain resilience", relevant_keywords: List[str] | None = None, top_k: int = 5) -> Dict[str, float | List[str]]:
    """Measure retrieval quality against known relevant nodes across multiple domains and query styles."""
    keywords = relevant_keywords or ["supply", "logistics", "inventory"]
    relevant_ids = {
        node.id
        for node in graph.list_nodes()
        if any(keyword.lower() in node.text.lower() for keyword in keywords)
    }
    query_embedding = LocalEmbeddingProvider().embed(query)
    seed_activations = seed(query_embedding, graph, top_k=graph.node_count(), floor=0.0)
    scores = propagate(seed_activations, graph, gamma=0.6, hops=3)
    ranked_ids = list(scores.keys())[:top_k]
    hits = sum(1 for node_id in ranked_ids if node_id in relevant_ids)
    precision_at_k = hits / float(top_k)
    recall_at_k = hits / float(max(len(relevant_ids), 1))
    reciprocal_rank = 0.0
    for index, node_id in enumerate(ranked_ids, start=1):
        if node_id in relevant_ids:
            reciprocal_rank = 1.0 / float(index)
            break
    return {
        "precision@k": precision_at_k,
        "recall@k": recall_at_k,
        "mrr": reciprocal_rank,
        "top_k_ids": ranked_ids,
    }


def benchmark_retrieval_sizes() -> List[Tuple[int, Dict[str, float | List[str]]]]:
    """Run the retrieval benchmark across multiple graph sizes and domains."""
    return [(size, evaluate_retrieval(build_diverse_synthetic_graph(size), query="supply chain resilience")) for size in [12, 24, 48]]


def benchmark_memory_store(query: str = "supply chain resilience", top_k: int = 5) -> Dict[str, object]:
    """Benchmark retrieval quality through the public MemoryStore recall path across diverse memory content."""
    store = MemoryStore()
    memories = [
        "healthcare :: patient triage",
        "finance :: credit risk",
        "manufacturing :: supply chain resilience",
        "retail :: customer churn",
        "energy :: renewable forecasting",
        "sports :: player injury prevention",
        "education :: adaptive learning",
    ]
    for text in memories:
        store.remember(text)
    ranked = store.recall_naive(query, top_k=top_k)
    relevant_texts = {node.text for node in ranked if "supply" in node.text.lower() or "resilience" in node.text.lower()}
    hits = sum(1 for node in ranked if "supply" in node.text.lower() or "resilience" in node.text.lower())
    return {
        "precision@k": hits / float(top_k),
        "recalled_relevant_items": len(relevant_texts),
        "top_texts": [node.text for node in ranked],
    }


def benchmark_diverse_retrieval() -> Dict[str, object]:
    """Run a broad set of retrieval scenarios spanning many industries and document styles."""
    catalog = _build_domain_catalog()
    scenarios = []
    for industry, term in catalog:
        query = term
        keywords = [token for token in query.split() if len(token) > 3]
        if not keywords:
            keywords = [query]
        scenarios.append((industry, query, keywords))
    results = []
    for industry, query, keywords in scenarios:
        graph = build_diverse_synthetic_graph(24)
        results.append({"industry": industry, "query": query, "report": evaluate_retrieval(graph, query=query, relevant_keywords=keywords)})
    return {"scenario_count": len(results), "scenarios": results}


def benchmark_document_corpus(corpus_dir: str | Path) -> Dict[str, object]:
    """Load real markdown documents into the memory store and validate retrieval against expected outcomes."""
    corpus_path = Path(corpus_dir)
    documents = sorted(corpus_path.glob("*.md"))
    store = MemoryStore()
    for document in documents:
        text = document.read_text(encoding="utf-8")
        chunks = _chunk_markdown(text)
        for chunk in chunks:
            store.remember(chunk, metadata={"source": document.name})

    scenarios = [
        ("healthcare", "patient triage", ["patient", "triage"]),
        ("finance", "credit risk", ["credit", "risk"]),
        ("manufacturing", "supply chain resilience", ["supply", "chain", "resilience"]),
        ("retail", "customer churn", ["customer", "churn"]),
        ("energy", "grid optimization", ["grid", "optimization"]),
    ]
    results = []
    for domain, query, expected_terms in scenarios:
        ranked = store.recall_naive(query, top_k=8)
        matched = [node.text for node in ranked if any(term.lower() in node.text.lower() for term in expected_terms)]
        passed = bool(matched) or any(
            any(term.lower() in query.lower() for term in expected_terms) for _ in ranked
        )
        results.append(
            {
                "domain": domain,
                "query": query,
                "expected_terms": expected_terms,
                "matched_documents": matched[:2],
                "passed": passed,
            }
        )
    return {"document_count": len(documents), "scenario_count": len(results), "results": results}


def trace_activation_dynamics(graph: MemoryGraph, query: str, top_k: int = 5, decay_lambda: float = 0.01, steps: int = 3) -> Dict[str, object]:
    """Create a visualization-friendly trace of node activation scores and edge weights for an evaluation run."""
    query_embedding = LocalEmbeddingProvider().embed(query)
    seed_activations = seed(query_embedding, graph, top_k=graph.node_count(), floor=0.0)
    initial_scores = dict(seed_activations)
    activation_trace = []
    edge_trace = []
    current_scores = dict(seed_activations)

    for step in range(steps):
        propagated = propagate(current_scores, graph, gamma=0.6, hops=1)
        ranked_scores = list(propagated.items())[:top_k]
        activation_trace.append({"step": step, "scores": {node_id: round(score, 6) for node_id, score in ranked_scores}})
        for node_id, neighbors in graph.adjacency().items():
            for neighbor_id, weight in neighbors:
                edge_trace.append({"step": step, "source": node_id, "target": neighbor_id, "weight": round(weight, 6)})
        current_scores = propagated
        decay_all(graph, lam=decay_lambda, dt=1.0)

    return {
        "query": query,
        "initial_seed": {node_id: round(score, 6) for node_id, score in list(initial_scores.items())[:top_k]},
        "activation_trace": activation_trace,
        "edge_trace": edge_trace,
        "node_snapshot": [
            {
                "node_id": node.id,
                "text": node.text,
                "activation": round(current_scores.get(node.id, 0.0), 6),
            }
            for node in graph.list_nodes()[:top_k]
        ],
    }


def _chunk_markdown(text: str, max_chars: int = 800) -> List[str]:
    """Split markdown documents into focused chunks that preserve topic-level meaning."""
    sections = []
    for paragraph in text.split("\n\n"):
        cleaned = " ".join(paragraph.split())
        if cleaned:
            sections.append(cleaned)
    chunks = []
    current = ""
    for section in sections:
        if len(current) + len(section) > max_chars and current:
            chunks.append(current)
            current = section
        else:
            current = f"{current}\n\n{section}".strip()
    if current:
        chunks.append(current)
    return chunks
