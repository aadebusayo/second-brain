# secondbrain

secondbrain is a graph-based associative memory package for LLM agents and other knowledge-driven applications.

## Installation

```bash
pip install secondbrain
```

## Quick start

```python
from secondbrain import remember, recall

remember("A note about market entry strategy")
results = recall("market entry")
print([node.text for node in results])
```

## CLI usage

```bash
secondbrain status
secondbrain remember "A note about market entry strategy"
secondbrain recall "market entry" --top-k 3
```

## Design goals

- pluggable embeddings
- pluggable vector stores
- durable storage for production use
- explainable retrieval
- extensible consolidation workflow

## Configuration

The package reads provider configuration from environment variables:

- EMBEDDING_PROVIDER
- VECTOR_PROVIDER
- STORAGE_BACKEND
- SECOND_BRAIN_DB_PATH
- SECOND_BRAIN_LOG_LEVEL

Example:

```bash
export EMBEDDING_PROVIDER=local
export VECTOR_PROVIDER=inmemory
export STORAGE_BACKEND=sqlite
export SECOND_BRAIN_DB_PATH=./data/secondbrain.sqlite
export SECOND_BRAIN_LOG_LEVEL=INFO
```

## Production notes

- The runtime exposes a status() method on MemoryStore for health-style checks.
- Persistence is durable through the SQLModel-backed storage layer.
- Unsupported providers fail fast with explicit configuration errors.
- Retrieval traces and timing metadata are emitted through the logger for debugging and observability.
