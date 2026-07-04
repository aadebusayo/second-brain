# API reference

## Public package entry points

### remember(text, metadata=None)
Store a new memory node in the package.

### recall(query, token_budget=2000)
Return top matching nodes for a user query.

### recall_naive(query, top_k=3)
Return top matching nodes with a naive cosine similarity ranked search.

### explain(node_id)
Return a lightweight explanation payload for a node's retrieval trace.

### consolidate()
Trigger a consolidation pass for the current memory graph.
