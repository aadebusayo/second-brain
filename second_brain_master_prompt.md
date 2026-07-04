# Second brain — master build prompt

**How to use this document.** Section 7 is the actual prompt — paste it into Claude Code, a Cowork session, or any coding agent to kick off the build. Sections 1-6 are the reasoning behind it: read them once, edit anything that doesn't match your setup (embedding provider, scale, budget), then hand the whole file to the agent. It's written so the agent can build phase by phase and check its own work against the acceptance criteria.

---

## 1. What this system is

A persistent associative memory layer for an LLM agent. Not a flat vector store — a weighted graph of memory nodes where retrieval works by activation spreading from a query seed outward through real, reinforced associations, gated by community/cluster structure, and periodically consolidated (pruned, summarized, re-clustered) the way episodic traces get compressed into semantic knowledge during sleep.

Package name used throughout: `secondbrain`. Rename as you like.

## 2. Architecture recap

- **Node** = an embedded chunk (a note, a fact, a conversation turn, a summary). Carries: embedding vector, text, metadata, access log (timestamps of retrieval), base-level activation.
- **Edge** = a weighted association between two nodes. Initial weight from cosine similarity; weight is then reinforced by co-retrieval (Hebbian: nodes recalled together get a stronger edge) and decays with time if unused.
- **Cluster** = a community of densely connected nodes, found by community detection, not hand-assigned. Clusters get their own centroid and inter-cluster edges, enabling two-stage retrieval (find the right neighborhood, then the right node).
- **Retrieval** = query embedding seeds activation on the nearest nodes → damped spreading activation propagates through weighted edges for a fixed number of hops → base-level activation (recency/frequency) is added in → top-N nodes under a token budget are assembled into context.
- **Consolidation** = a background job: decays stale edges, promotes frequently co-activated subgraphs into a single LLM-written summary node, re-runs community detection.

## 3. Component inventory — leverage, don't rebuild

| Layer | Use | Don't build |
|---|---|---|
| Embeddings | Anthropic or a local `sentence-transformers` model | Your own embedding model |
| Vector / ANN search | `LanceDB` or `chromadb` (both embed HNSW, which is already a hierarchical graph index) | A custom nearest-neighbor index |
| Graph substrate | `networkx` for prototyping; `kuzu` (embedded, Cypher-capable) or `neo4j` once it outgrows memory | A custom graph database |
| Community detection | `python-igraph` + `leidenalg` (Leiden), or `networkx.algorithms.community` (Louvain) | A custom clustering algorithm |
| Metadata / access log persistence | `SQLite` via `sqlmodel` | A custom persistence layer |
| Scheduling consolidation jobs | `APScheduler` | A custom job scheduler |
| Summarization calls during consolidation | Anthropic SDK, `claude-sonnet-5` or similar | — |
| Exposing this to an agent harness | `mcp` (Model Context Protocol Python SDK) — turn the package's public API into MCP tools | A custom RPC/tool protocol |
| Testing | `pytest` | — |

Verdict: roughly 60% of this system is integration of mature, boring libraries. That's good — it means the novel 40% is where the actual value is, and it's small enough to build and test properly.

## 4. Novel components — this is the actual system

These don't exist off the shelf. Each is a module with a narrow, testable contract.

### `secondbrain.weights` — the Hebbian reinforcement engine
```
reinforce(graph, i, j, eta=0.1) -> None
    # w(i,j) += eta * (co_activation - w(i,j))
    # called whenever i and j are retrieved together in the same recall
decay_all(graph, lam=0.01, dt=1.0) -> None
    # w(i,j) *= exp(-lam * dt) for every edge, on a schedule
    # edges below a floor threshold get dropped
```

### `secondbrain.activation` — spreading activation + base-level activation
```
base_level(node, decay=0.5) -> float
    # ACT-R style: ln(sum(t_k ** -decay for t_k in node.access_log))
    # rewards both recency and frequency

seed(query_embedding, graph, top_k=8, floor=0.2) -> dict[node_id, float]
    # cosine similarity seeding, thresholded so irrelevant nodes get 0

propagate(seed_activations, graph, gamma=0.6, hops=3) -> dict[node_id, float]
    # damped spread: a_i(h+1) = a_i(0) + B(i) + gamma * sum(w_ij * a_j(h))
    # normalize each hop to prevent runaway growth
    # THIS IS THE CORE ALGORITHM. Everything else is plumbing around it.
```

### `secondbrain.clusters` — two-stage retrieval
```
detect_communities(graph) -> dict[node_id, cluster_id]
    # wraps leidenalg; run after every consolidation pass

gate_clusters(query_embedding, cluster_centroids, top_c=3) -> list[cluster_id]
    # cheap first pass: which neighborhoods are even relevant

drill_down(query_embedding, graph, cluster_ids) -> dict[node_id, float]
    # only run full spreading activation inside the gated clusters
    # this is what keeps retrieval cheap as the graph grows
```

### `secondbrain.assemble` — context-budget selection
```
select(activations, token_counts, budget) -> list[node_id]
    # greedy knapsack: rank by activation / token_count, fill the budget
    # this is where "everything activated" becomes "what actually fits"
```

### `secondbrain.consolidate` — the "sleep" job
```
find_consolidation_candidates(graph, min_cluster_size=4, min_avg_weight=0.5) -> list[list[node_id]]
    # dense, frequently co-activated subgraphs

summarize_subgraph(nodes, llm_client) -> Node
    # one LLM call per candidate, writes a new compressed summary node

rewire(graph, old_nodes, summary_node) -> None
    # point external edges at the summary node
    # demote (don't delete) the originals — keep provenance, lower their base activation
```

### `secondbrain.api` — the public surface (this is what the agent harness sees)
```
remember(text, metadata=None) -> node_id
recall(query, token_budget=2000) -> list[Node]
explain(node_id) -> dict
    # returns the actual activation math that surfaced this node —
    # non-negotiable for debuggability and user trust. Don't skip this.
consolidate() -> ConsolidationReport
```

### `secondbrain.eval` — retrieval quality harness
There is no standard benchmark for personal associative memory. You'll build a small synthetic one: seed a graph with known ground-truth associations, run queries, measure whether the intended nodes surface and whether irrelevant nodes stay suppressed. Track this over time — it's the only way to know if a change to `gamma` or `eta` helped or hurt.

## 5. Package layout

```
secondbrain/
  pyproject.toml
  src/secondbrain/
    __init__.py
    store.py          # vector DB + SQLite metadata wrapper
    graph.py           # networkx/kuzu wrapper: add_node, add_edge, get_neighbors
    weights.py          # Hebbian reinforcement + decay
    activation.py        # seed, propagate, base_level
    clusters.py          # community detection, two-stage retrieval
    assemble.py          # token-budget knapsack
    consolidate.py        # sleep job
    api.py             # public functions: remember, recall, explain, consolidate
    mcp_server.py        # wraps api.py as MCP tools
  tests/
    test_activation.py
    test_weights.py
    test_consolidate.py
    test_eval_harness.py
```

## 6. Build order

1. `store` + `graph` + naive cosine-only recall (no activation yet) — prove the plumbing works end to end.
2. `activation` — add spreading activation on top of step 1's graph. This is the highest-risk module; get the eval harness running against it before moving on.
3. `clusters` — two-stage retrieval for scale.
4. `weights` + `consolidate` — reinforcement, decay, the sleep job.
5. `mcp_server` — expose to an agent harness.
6. `eval` — formalize and keep running as a regression check on every change to steps 2-4.

Do not build 3-6 before 1-2 are passing their tests. The spreading activation algorithm is the part most likely to need tuning (gamma, hop count, decay rate) — it needs to be isolated and testable before anything else depends on it.

---

## 7. The prompt — paste this to your coding agent

```
You are building a Python package called `secondbrain`: a persistent,
graph-based associative memory system for LLM agents, implementing
damped spreading activation retrieval over a weighted node/edge graph
with periodic consolidation.

Build it in the six phases below, in order. After each phase, run its
tests and show me the results before moving to the next phase — do not
build ahead of what's been verified.

Use these libraries, do not reimplement them:
- Embeddings: Anthropic API (or sentence-transformers if no API key is
  configured)
- Vector search: LanceDB
- Graph substrate: networkx to start
- Community detection: leidenalg via python-igraph
- Metadata persistence: SQLite via sqlmodel
- Scheduling: APScheduler
- Agent-harness exposure: the official `mcp` Python SDK

Package layout, module responsibilities, and exact function signatures
for the novel components (weights, activation, clusters, assemble,
consolidate, api) are specified in sections 4-5 of the attached spec.
Follow those signatures exactly so the modules stay independently
testable.

Phase 1 — store + graph + naive recall
  Build store.py and graph.py. Wire a `remember(text)` /
  `recall_naive(query)` path using plain cosine similarity only, no
  activation spreading yet. Acceptance: I can remember 20 short notes
  and recall the 3 most similar to a query.

Phase 2 — spreading activation
  Build activation.py exactly per the seed/propagate/base_level
  signatures in section 4. Build tests/test_activation.py with a small
  hand-constructed graph where you know the correct activation ranking
  in advance (e.g. a seed node, a strongly-weighted neighbor, a
  weakly-weighted two-hop node, and a disconnected node) and assert the
  output ranks them correctly. Acceptance: activation ranking matches
  hand-computed expected values within a small tolerance.

Phase 3 — clusters
  Build clusters.py. Wire gate_clusters + drill_down into recall so
  full spreading activation only runs inside gated clusters. Acceptance:
  recall on a 200+ node synthetic graph completes without running
  propagation on the whole graph every time.

Phase 4 — weights + consolidate
  Build weights.py (Hebbian reinforcement on co-retrieval, time decay)
  and consolidate.py (find dense co-activated subgraphs, call the LLM
  to summarize them into a new node, rewire edges, demote originals
  without deleting them). Acceptance: after simulating repeated
  co-retrieval of two nodes, their edge weight measurably increases;
  after running consolidate() on a dense cluster, a new summary node
  exists and the original nodes' base activation has dropped but they
  still exist.

Phase 5 — MCP exposure
  Build mcp_server.py wrapping api.py's remember / recall / explain /
  consolidate as MCP tools. explain(node_id) must return the actual
  activation math (seed value, hop-by-hop contributions, final score)
  that justified surfacing that node, not just a text description.
  Acceptance: the MCP server starts and all four tools are callable
  from an MCP client.

Phase 6 — eval harness
  Build eval.py: a synthetic graph with known ground-truth associations
  and a set of queries with expected top-k results. Report precision@k
  and a measure of how much irrelevant activation leaks into results.
  Run this after phases 2-4 to catch regressions if gamma, eta, or
  decay rates change later.

Do not add a UI, a CLI, or authentication — this is a library meant to
be imported by an agent harness or wrapped by the MCP server, nothing
else. Do not add configuration options beyond what's needed to tune
gamma, eta, decay rate, and hop count — those four are the ones that
will actually need tuning against real usage.
```
