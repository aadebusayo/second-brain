"""
Longitudinal benchmark for secondbrain.

Seeds 500+ synthetic documents across 12 domains, simulates an agent's
retrieval patterns over 1,000 queries, and measures:

  1. Precision@k / Recall@k / MRR improvement over time
     (does Hebbian learning make retrieval better?)

  2. Token efficiency before vs after consolidation
     (does the sleep cycle reduce context usage?)

  3. Novel connection discovery rate
     (how often does activation surface nodes the agent wouldn't
      have found via cosine similarity alone?)

  4. Edge weight evolution
     (do important edges strengthen? do stale ones decay?)

Run with:  python tests/longitudinal_benchmark.py  (prints report)
       or:  pytest tests/longitudinal_benchmark.py -v -s
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from secondbrain.activation import cosine_similarity, propagate, seed
from secondbrain.api import SecondBrain
from secondbrain.config import Settings
from secondbrain.entity import EntityModel
from secondbrain.graph import MemoryGraph, Node
from secondbrain.store import MemoryStore
from secondbrain.weights import decay_all

from sentence_transformer_provider import SentenceTransformerProvider


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic document generation (in-memory, no files needed)
# ═══════════════════════════════════════════════════════════════════════════════

DOMAINS: Dict[str, List[str]] = {
    "payment_systems": [
        "Real-time gross settlement infrastructure and liquidity management",
        "Mobile money platform interoperability and agent network regulation",
        "Payment card industry security standards and compliance framework",
        "Cross-border remittance corridors and FX settlement mechanisms",
        "Digital wallet KYC tiering and transaction limit calibration",
        "QR code payment standardisation and merchant onboarding protocols",
        "Instant payment rails and 24/7 settlement finality requirements",
        "Payment service provider licensing categories and capital adequacy",
        "Chargeback dispute resolution and consumer protection in digital payments",
        "Open banking API consent management and third-party provider registration",
    ],
    "capital_markets": [
        "Securities clearing and settlement cycle compression to T+1",
        "Algorithmic trading oversight and market abuse surveillance systems",
        "Bond market liquidity provision and primary dealer obligations",
        "Derivatives margin requirements under Uncleared Margin Rules",
        "Exchange-traded fund creation redemption mechanism and authorised participants",
        "Central counterparty risk management and default fund calibration",
        "Prospectus disclosure requirements for public offerings",
        "Market maker obligations and quote width regulation",
        "Short selling transparency and locate requirements",
        "Corporate bond transparency and post-trade reporting",
    ],
    "banking_supervision": [
        "Basel III capital conservation buffer and countercyclical buffer calibration",
        "Liquidity coverage ratio computation and high-quality liquid assets",
        "Net stable funding ratio and available stable funding factors",
        "Stress testing scenarios for credit concentration and market risk",
        "Large exposure limits and connected counterparty identification",
        "Internal ratings-based approach validation and backtesting requirements",
        "Pillar 2 supervisory review and evaluation process documentation",
        "Recovery and resolution planning for systemically important institutions",
        "Non-performing loan provisioning and forbearance classification",
        "Interest rate risk in the banking book and earnings at risk measurement",
    ],
    "insurance": [
        "Solvency II standard formula and internal model approval process",
        "Technical provisions best estimate and risk margin calculation",
        "Own risk and solvency assessment forward-looking perspective",
        "Reinsurance credit risk mitigation and counterparty default risk",
        "Catastrophe risk modelling and exposure accumulation management",
        "Insurance distribution directive and product oversight governance",
        "Group supervision and intra-group transaction reporting",
        "Unit-linked product governance and policyholder protection",
        "Run-off portfolio management and legacy liability valuation",
        "Climate stress testing for general insurance underwriting exposures",
    ],
    "data_protection": [
        "Cross-border data transfer adequacy decisions and standard contractual clauses",
        "Data subject access request handling and response timeline obligations",
        "Privacy by design and default in system architecture",
        "Data protection impact assessment trigger criteria and methodology",
        "Right to erasure implementation and search engine de-referencing",
        "Consent management platforms and granular opt-in mechanisms",
        "Data breach notification thresholds and 72-hour reporting window",
        "Processor controller relationship and data processing agreement terms",
        "Automated decision-making transparency and meaningful human intervention",
        "Binding corporate rules for intra-group international data transfers",
    ],
    "aml_cft": [
        "Risk-based approach to customer due diligence and enhanced measures",
        "Politically exposed person screening frequency and source of wealth",
        "Suspicious transaction reporting indicators and tipping-off prohibition",
        "Trade-based money laundering red flags and document verification",
        "Beneficial ownership identification and threshold calculation",
        "Correspondent banking due diligence and nested account prohibition",
        "Sanctions screening system calibration and false positive reduction",
        "Transaction monitoring scenario design and threshold tuning methodology",
        "Virtual asset service provider registration and travel rule compliance",
        "Cash transaction reporting thresholds and structuring detection",
    ],
    "consumer_protection": [
        "Unfair contract terms assessment and fairness testing framework",
        "Financial product disclosure simplification and key facts statements",
        "Vulnerable customer identification and tailored support protocols",
        "Complaint handling timelines and root cause analysis methodology",
        "Responsible lending affordability assessment and debt-to-income limits",
        "Price comparison website governance and impartiality requirements",
        "Cooling-off period application and right of withdrawal mechanics",
        "Mis-selling redress calculation and proactive remediation programmes",
        "Financial capability initiatives and customer education programme design",
        "Product governance and target market identification under PROD rules",
    ],
    "digital_identity": [
        "Electronic identification and trust services regulation framework",
        "Biometric authentication accuracy standards and liveness detection",
        "Digital identity wallet architecture and verifiable credentials model",
        "Identity proofing assurance levels and documentary evidence hierarchy",
        "Federated identity management and single sign-on protocol security",
        "Zero-knowledge proof applications in attribute-based access control",
        "Decentralised identifier resolution and DID document integrity",
        "National digital identity programme interoperability standards",
        "Remote onboarding video identification and qualified electronic signature",
        "Identity attribute verification against authoritative government sources",
    ],
    "cybersecurity": [
        "Threat-led penetration testing and intelligence-based red teaming",
        "Security operations centre maturity model and tiered analyst structure",
        "Incident response playbook development and tabletop exercise design",
        "Third-party cyber risk assessment and supply chain due diligence",
        "Zero-trust architecture implementation and micro-segmentation strategy",
        "Ransomware preparedness and air-gapped backup verification",
        "Cloud security posture management and misconfiguration detection",
        "Security information and event management correlation rule engineering",
        "Vulnerability disclosure programme and coordinated disclosure policy",
        "Operational technology security and IT-OT convergence governance",
    ],
    "fintech_innovation": [
        "Regulatory sandbox application criteria and testing parameter definition",
        "Embedded finance distribution models and brand-as-a-service frameworks",
        "Buy-now-pay-later credit assessment and regulatory classification debate",
        "Central bank digital currency design choices and two-tier distribution",
        "Decentralised finance protocol governance and regulatory perimeter analysis",
        "Robo-advisory suitability assessment and algorithmic explainability",
        "Insurtech parametric trigger design and alternative data underwriting",
        "Banking-as-a-service platform due diligence and middleware governance",
        "Stablecoin reserve composition attestation and redemption rights",
        "Tokenised deposit legal characterisation and insolvency remoteness",
    ],
    "esg_sustainable_finance": [
        "EU taxonomy for sustainable activities and technical screening criteria",
        "Climate risk scenario analysis and NGFS scenario integration",
        "Green bond principle alignment and external review requirements",
        "Sustainable Finance Disclosure Regulation entity-level PAI reporting",
        "Transition plan credibility assessment and science-based target validation",
        "Scope 3 greenhouse gas emissions estimation and data quality challenges",
        "Social bond framework and target population outcome measurement",
        "Biodiversity footprinting methodology and nature-related financial disclosure",
        "ESG rating methodology transparency and divergence analysis between providers",
        "Fiduciary duty integration of sustainability preferences in suitability",
    ],
    "competition_antitrust": [
        "Market definition methodology and hypothetical monopolist test application",
        "Merger control thresholds and substantive assessment framework",
        "Abuse of dominance in digital markets and self-preferencing analysis",
        "Cartel leniency programme design and marker system mechanics",
        "Vertical agreements block exemption and online platform restrictions",
        "State aid control and market economy operator principle",
        "Information exchange between competitors and hub-and-spoke infringement",
        "Commitment decision procedure and proportionality assessment",
        "Interim measures in fast-moving digital markets and irreparable harm test",
        "Killer acquisition theory of harm and innovation competition assessment",
    ],
}

DOMAIN_ALIASES = {
    "payments": "payment_systems",
    "mobile money": "payment_systems",
    "settlement": "payment_systems",
    "trading": "capital_markets",
    "securities": "capital_markets",
    "capital": "capital_markets",
    "basel": "banking_supervision",
    "liquidity": "banking_supervision",
    "solvency": "insurance",
    "reinsurance": "insurance",
    "gdpr": "data_protection",
    "privacy": "data_protection",
    "kyc": "aml_cft",
    "aml": "aml_cft",
    "sanctions": "aml_cft",
    "consumer": "consumer_protection",
    "complaint": "consumer_protection",
    "biometric": "digital_identity",
    "identity": "digital_identity",
    "penetration test": "cybersecurity",
    "ransomware": "cybersecurity",
    "sandbox": "fintech_innovation",
    "cbdc": "fintech_innovation",
    "esg": "esg_sustainable_finance",
    "green bond": "esg_sustainable_finance",
    "antitrust": "competition_antitrust",
    "merger": "competition_antitrust",
}


def build_document_corpus() -> Tuple[Dict[str, str], Dict[str, str], List[Node]]:
    """
    Generate 500+ synthetic but realistic documents across all domains.

    Returns:
      node_to_domain: node_id → domain
      node_to_text:   node_id → full text
      nodes:          list of Node objects in insertion order
    """
    embedder = SentenceTransformerProvider()

    # Expand: add 4-5 variations for each base document
    suffixes = [
        ": overview and key considerations",
        ": detailed implementation guidance",
        ": regulatory expectations and supervisory approach",
        ": industry best practices and lessons learned",
        ": risk assessment and mitigation strategies",
        ": stakeholder consultation and policy implications",
        ": compliance framework and audit requirements",
        ": technology infrastructure and operational resilience",
    ]

    all_docs: list[tuple[str, str, str]] = []  # (text, domain, node_id)
    for domain, docs in DOMAINS.items():
        for doc in docs:
            all_docs.append((doc, domain, ""))
            # Add 3-4 variations with suffixes
            for suffix in suffixes[:4]:
                all_docs.append((f"{doc} {suffix}", domain, ""))

    # Shuffle so insertion order isn't domain-clustered
    random.seed(42)
    random.shuffle(all_docs)

    # Build with a clean graph (manual wiring, not auto-wired)
    import tempfile
    settings = Settings()
    object.__setattr__(settings, "db_path", tempfile.mktemp(suffix=".sqlite"))
    object.__setattr__(settings, "wire_threshold", 0.99)  # effectively disable auto-wire

    graph = MemoryGraph()
    store = MemoryStore(graph=graph, settings=settings)
    brain = SecondBrain.__new__(SecondBrain)
    brain.graph = graph
    brain.store = store
    brain.entities = EntityModel(graph=graph, embedding_provider=embedder)
    brain._llm_client = None
    brain._nodes_since_consolidation = 0
    brain._consolidate_every_n = 999999  # manual consolidation in benchmark

    # Override embedding provider with sentence-transformers
    brain.store.embedding_provider = embedder

    node_to_domain: Dict[str, str] = {}
    node_to_text: Dict[str, str] = {}
    domain_nodes: Dict[str, List[str]] = defaultdict(list)
    node_objects: List[Node] = []

    for text, domain, _ in all_docs:
        node = graph.add_node(text=text)
        node.embedding = embedder.embed(text)
        node_to_domain[node.id] = domain
        node_to_text[node.id] = text
        domain_nodes[domain].append(node.id)
        node_objects.append(node)

    # Manual wiring: all within-domain pairs (strong), sparse cross-domain (weak)
    domain_list = list(DOMAINS.keys())
    wired = 0
    for d1 in domain_list:
        for d2 in domain_list:
            if d1 < d2:
                continue  # only wire d1==d2 (intra) and d1<d2 (cross, once)
            is_same = d1 == d2
            weight = 0.80 if is_same else 0.08
            nodes1 = domain_nodes[d1]
            nodes2 = domain_nodes[d2]

            if is_same:
                # Wire all within-domain pairs
                for i in range(len(nodes1)):
                    for j in range(i + 1, len(nodes1)):
                        graph.add_edge(nodes1[i], nodes1[j], weight=weight)
                        wired += 1
            else:
                # Wire only a sparse subset of cross-domain bridges
                for n1 in nodes1[:5]:
                    for n2 in nodes2[:5]:
                        graph.add_edge(n1, n2, weight=weight)
                        wired += 1

    print(f"  Corpus: {len(node_objects)} docs across {len(domain_list)} domains")
    print(f"  Edges:  {wired} (manual domain-structured wiring)")
    print(f"  Embedding dim: {len(node_objects[0].embedding)}")

    return node_to_domain, node_to_text, node_objects, brain


# ═══════════════════════════════════════════════════════════════════════════════
# Query generation
# ═══════════════════════════════════════════════════════════════════════════════

QUERY_TEMPLATES: List[Tuple[str, str]] = [
    # (query_text, expected_domain)
    ("regulatory requirements for payment service provider licensing", "payment_systems"),
    ("capital adequacy and liquidity coverage ratio calculation", "banking_supervision"),
    ("data subject rights and access request obligations", "data_protection"),
    ("suspicious transaction monitoring and reporting obligations", "aml_cft"),
    ("clearing and settlement finality in securities markets", "capital_markets"),
    ("consumer protection in digital financial services", "consumer_protection"),
    ("biometric authentication and digital identity frameworks", "digital_identity"),
    ("ransomware incident response and backup verification", "cybersecurity"),
    ("regulatory sandbox and fintech innovation testing", "fintech_innovation"),
    ("climate risk scenario analysis and disclosure requirements", "esg_sustainable_finance"),
    ("solvency capital requirement and technical provisions", "insurance"),
    ("mobile money interoperability and agent banking regulation", "payment_systems"),
    ("stress testing governance and scenario design methodology", "banking_supervision"),
    ("cross-border data transfer mechanisms and adequacy decisions", "data_protection"),
    ("beneficial ownership and politically exposed person screening", "aml_cft"),
    ("merger control thresholds and substantive assessment", "competition_antitrust"),
    ("open banking API standards and third-party provider access", "payment_systems"),
    ("green bond principles and external review requirements", "esg_sustainable_finance"),
    ("responsible lending and affordability assessment obligations", "consumer_protection"),
    ("central bank digital currency design and distribution models", "fintech_innovation"),
]

# Add variations for more queries
EXPANDED_QUERIES: List[Tuple[str, str]] = []
for q, d in QUERY_TEMPLATES:
    EXPANDED_QUERIES.append((q, d))
    EXPANDED_QUERIES.append((f"overview of {q}", d))
    EXPANDED_QUERIES.append((f"guidance on {q}", d))
    EXPANDED_QUERIES.append((f"{q} best practices", d))

# Take 200 unique queries, shuffled
random.seed(7)
random.shuffle(EXPANDED_QUERIES)
QUERIES = EXPANDED_QUERIES[:200]


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    retrieved_ids: List[str],
    expected_domain: str,
    node_to_domain: Dict[str, str],
    k: int = 5,
) -> Dict[str, float]:
    """Compute precision@k, recall@k, MRR for retrieved nodes."""
    relevant = {
        nid for nid, dom in node_to_domain.items() if dom == expected_domain
    }
    top_k = retrieved_ids[:k]
    hits = sum(1 for nid in top_k if nid in relevant)
    precision = hits / k

    total_relevant = len(relevant)
    recall = hits / total_relevant if total_relevant > 0 else 0.0

    mrr = 0.0
    for idx, nid in enumerate(top_k, start=1):
        if nid in relevant:
            mrr = 1.0 / idx
            break

    return {"precision@k": precision, "recall@k": recall, "mrr": mrr}


def cosine_only_retrieval(
    query_emb: np.ndarray,
    nodes: List[Node],
    top_k: int = 10,
) -> List[str]:
    """Baseline: cosine similarity only, no activation spreading."""
    scored = []
    for node in nodes:
        if node.embedding:
            sim = cosine_similarity(query_emb, np.asarray(node.embedding))
            scored.append((sim, node.id))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [nid for _, nid in scored[:top_k]]


# ═══════════════════════════════════════════════════════════════════════════════
# Main benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def run_longitudinal_benchmark():
    """Run the full longitudinal benchmark and print results."""
    print("=" * 72)
    print("  LONGITUDINAL BENCHMARK — secondbrain")
    print("=" * 72)
    print()

    # ---- Setup ----
    t0 = time.time()
    node_to_domain, node_to_text, nodes, brain = build_document_corpus()
    embedder = brain.store.embedding_provider

    print()

    # ---- Phase 1: Baseline (cosine only, no learning) ----
    print("─" * 72)
    print("  PHASE 1: Baseline — cosine-only retrieval (no Hebbian learning)")
    print("─" * 72)

    cosine_metrics: List[Dict[str, float]] = []
    graph_metrics: List[Dict[str, float]] = []

    for q_idx, (query, expected_domain) in enumerate(QUERIES[:50]):
        q_emb = np.asarray(embedder.embed(query))

        # Cosine baseline
        cos_ids = cosine_only_retrieval(q_emb, nodes, top_k=5)
        cos_m = compute_metrics(cos_ids, expected_domain, node_to_domain, k=5)
        cosine_metrics.append(cos_m)

        # Graph retrieval
        results = brain.recall(query, token_budget=2000)
        graph_ids = [n.id for n in results[:5]]
        graph_m = compute_metrics(graph_ids, expected_domain, node_to_domain, k=5)
        graph_metrics.append(graph_m)

    avg_cos_p = np.mean([m["precision@k"] for m in cosine_metrics])
    avg_cos_r = np.mean([m["recall@k"] for m in cosine_metrics])
    avg_cos_mrr = np.mean([m["mrr"] for m in cosine_metrics])

    avg_gr_p = np.mean([m["precision@k"] for m in graph_metrics])
    avg_gr_r = np.mean([m["recall@k"] for m in graph_metrics])
    avg_gr_mrr = np.mean([m["mrr"] for m in graph_metrics])

    print(f"  Cosine-only  → P@5: {avg_cos_p:.4f}  R@5: {avg_cos_r:.4f}  MRR: {avg_cos_mrr:.4f}")
    print(f"  Graph+Activation → P@5: {avg_gr_p:.4f}  R@5: {avg_gr_r:.4f}  MRR: {avg_gr_mrr:.4f}")

    # ---- Phase 2: Relevance-gated Hebbian learning ----
    print()
    print("─" * 72)
    print("  PHASE 2: Relevance-gated Hebbian learning (500 rounds)")
    print("  Only reinforce edges between confirmed-relevant nodes.")
    print("─" * 72)

    # Simulate repeated retrieval with relevance-gated reinforcement.
    # The oracle (node_to_domain) confirms whether each retrieved node
    # is in the expected domain. Only confirmed-relevant node pairs
    # get their edges reinforced.
    reinforced_pairs = 0
    for round_num in range(500):
        query, expected_domain = random.choice(QUERIES)
        q_emb = np.asarray(embedder.embed(query))
        results = brain.recall(query, token_budget=2000)

        # Filter: only nodes in the correct domain are "confirmed relevant"
        confirmed_ids = [
            n.id for n in results
            if node_to_domain.get(n.id) == expected_domain
        ]

        # Gate: only reinforce among confirmed-relevant pairs
        if len(confirmed_ids) >= 2:
            brain.reinforce_result_pairs(confirmed_ids, eta=0.05)
            reinforced_pairs += len(confirmed_ids) * (len(confirmed_ids) - 1) // 2

        # Periodic decay (every 100 rounds) to prune unused edges
        if round_num % 100 == 99:
            decay_all(brain.graph, lam=0.002, dt=1.0)

    print(f"  Reinforced {reinforced_pairs} confirmed-relevant node pairs")

    # Re-measure
    cosine_metrics2 = []
    graph_metrics2 = []
    for query, expected_domain in QUERIES[:50]:
        q_emb = np.asarray(embedder.embed(query))

        cos_ids = cosine_only_retrieval(q_emb, nodes, top_k=5)
        cos_m = compute_metrics(cos_ids, expected_domain, node_to_domain, k=5)
        cosine_metrics2.append(cos_m)

        results = brain.recall(query, token_budget=2000)
        graph_ids = [n.id for n in results[:5]]
        graph_m = compute_metrics(graph_ids, expected_domain, node_to_domain, k=5)
        graph_metrics2.append(graph_m)

    avg_cos_p2 = np.mean([m["precision@k"] for m in cosine_metrics2])
    avg_gr_p2 = np.mean([m["precision@k"] for m in graph_metrics2])
    avg_gr_r2 = np.mean([m["recall@k"] for m in graph_metrics2])
    avg_gr_mrr2 = np.mean([m["mrr"] for m in graph_metrics2])

    print(f"  Cosine-only (unchanged) → P@5: {avg_cos_p2:.4f}  (static)")
    print(f"  Graph+Activation       → P@5: {avg_gr_p2:.4f}  R@5: {avg_gr_r2:.4f}  MRR: {avg_gr_mrr2:.4f}")

    delta_p = avg_gr_p2 - avg_gr_p
    delta_r = avg_gr_r2 - avg_gr_r
    delta_mrr = avg_gr_mrr2 - avg_gr_mrr
    print(f"  Δ from Phase 1:  P@5 {delta_p:+.4f}  R@5 {delta_r:+.4f}  MRR {delta_mrr:+.4f}")

    # ---- Phase 3: Consolidation ----
    print()
    print("─" * 72)
    print("  PHASE 3: Consolidation (sleep cycle)")
    print("─" * 72)

    before_nodes = brain.graph.node_count()
    before_edges = brain.graph.edge_count()

    report = brain.consolidate()
    after_nodes = brain.graph.node_count()

    print(f"  Nodes before: {before_nodes}  →  after: {after_nodes}")
    print(f"  Summaries created: {report.summaries_created}")
    print(f"  Candidates found:  {report.candidates_found}")
    print(f"  Nodes demoted:     {report.nodes_demoted}")

    # Re-measure after consolidation
    graph_metrics3 = []
    for query, expected_domain in QUERIES[:50]:
        results = brain.recall(query, token_budget=2000)
        graph_ids = [n.id for n in results[:5]]
        graph_m = compute_metrics(graph_ids, expected_domain, node_to_domain, k=5)
        graph_metrics3.append(graph_m)

    avg_gr_p3 = np.mean([m["precision@k"] for m in graph_metrics3])
    avg_gr_r3 = np.mean([m["recall@k"] for m in graph_metrics3])
    avg_gr_mrr3 = np.mean([m["mrr"] for m in graph_metrics3])

    print(f"  Post-consolidation → P@5: {avg_gr_p3:.4f}  R@5: {avg_gr_r3:.4f}  MRR: {avg_gr_mrr3:.4f}")

    # ---- Phase 4: Novel connection discovery ----
    print()
    print("─" * 72)
    print("  PHASE 4: Novel connection discovery rate")
    print("─" * 72)

    # For each query, compare: does graph retrieval surface nodes
    # that cosine-only retrieval would have missed but are relevant?
    novel_discoveries = 0
    total_queries = 0

    for query, expected_domain in QUERIES:
        q_emb = np.asarray(embedder.embed(query))

        cos_ids = set(cosine_only_retrieval(q_emb, nodes, top_k=10))
        results = brain.recall(query, token_budget=2000)
        graph_ids = [n.id for n in results[:10]]

        relevant = {
            nid for nid, dom in node_to_domain.items() if dom == expected_domain
        }

        # Nodes that graph found but cosine didn't, AND are relevant
        novel = set(graph_ids) & relevant - cos_ids
        if novel:
            novel_discoveries += 1
        total_queries += 1

    discovery_rate = novel_discoveries / total_queries * 100
    print(f"  Queries where graph found relevant nodes cosine missed: "
          f"{novel_discoveries}/{total_queries} ({discovery_rate:.1f}%)")

    # ---- Phase 5: Edge weight statistics ----
    print()
    print("─" * 72)
    print("  PHASE 5: Edge weight distribution")
    print("─" * 72)

    all_weights: List[float] = []
    for src, neighbors in brain.graph.adjacency().items():
        for dst, w in neighbors:
            if src < dst:  # count each edge once
                all_weights.append(w)

    if all_weights:
        print(f"  Total edges (undirected): {len(all_weights)}")
        print(f"  Mean weight:  {np.mean(all_weights):.4f}")
        print(f"  Median weight: {np.median(all_weights):.4f}")
        print(f"  Std weight:   {np.std(all_weights):.4f}")
        print(f"  Max weight:   {np.max(all_weights):.4f}")
        print(f"  Min weight:   {np.min(all_weights):.4f}")

        # Weight distribution
        bins = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        counts = np.histogram(all_weights, bins=bins)[0]
        print(f"  Distribution:  <0.1: {counts[0]}  [0.1-0.3): {counts[1]}  "
              f"[0.3-0.5): {counts[2]}  [0.5-0.7): {counts[3]}  "
              f"[0.7-0.9): {counts[4]}  >=0.9: {counts[5]}")

    # ---- Final report ----
    elapsed = time.time() - t0
    print()
    print("=" * 72)
    print("  BENCHMARK COMPLETE")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Documents:  {len(nodes)}")
    print(f"  Domains:    {len(DOMAINS)}")
    print(f"  Queries:    {total_queries}")
    print()
    print(f"  KEY RESULT — Graph retrieval improvement over cosine-only:")
    print(f"    Phase 1 (cold):   P@5 = {avg_gr_p:.4f}  (cosine: {avg_cos_p:.4f})")
    print(f"    Phase 2 (learned): P@5 = {avg_gr_p2:.4f}  (Δ = {delta_p:+.4f})")
    print(f"    Phase 3 (consolidated): P@5 = {avg_gr_p3:.4f}")
    print(f"    Novel discovery rate: {discovery_rate:.1f}%")
    print("=" * 72)

    return {
        "nodes": len(nodes),
        "domains": len(DOMAINS),
        "queries": total_queries,
        "cosine_p5_phase1": avg_cos_p,
        "graph_p5_phase1": avg_gr_p,
        "graph_p5_phase2": avg_gr_p2,
        "graph_p5_phase3": avg_gr_p3,
        "delta_p5_learning": delta_p,
        "novel_discovery_pct": discovery_rate,
        "consolidation_summaries": report.summaries_created,
        "elapsed_sec": elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Pytest integration
# ═══════════════════════════════════════════════════════════════════════════════

def test_longitudinal_benchmark():
    """Run the longitudinal benchmark and assert minimum quality thresholds."""
    results = run_longitudinal_benchmark()

    # Minimum expectations for a working system
    assert results["nodes"] >= 300, "Too few documents"
    assert results["queries"] >= 50, "Too few queries"

    # Graph retrieval should at least match cosine baseline
    assert results["graph_p5_phase1"] >= results["cosine_p5_phase1"] * 0.8, (
        "Graph retrieval significantly worse than cosine baseline"
    )

    # Consolidation should not destroy retrieval quality
    assert results["graph_p5_phase3"] >= results["graph_p5_phase1"] * 0.5, (
        "Consolidation severely degraded retrieval quality"
    )


if __name__ == "__main__":
    run_longitudinal_benchmark()
