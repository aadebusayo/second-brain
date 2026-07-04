# Production deployment notes

## Environment configuration

Use environment variables to select runtime behavior explicitly:

```bash
export EMBEDDING_PROVIDER=local
export VECTOR_PROVIDER=inmemory
export STORAGE_BACKEND=sqlite
export SECOND_BRAIN_DB_PATH=./data/secondbrain.sqlite
export SECOND_BRAIN_LOG_LEVEL=INFO
```

## Operational checks

- Inspect runtime status with MemoryStore.status().
- Review logs for remember and recall events.
- Persisted data lives in the configured SQLite database path.

## Scaling guidance

- For larger graph workloads, prefer a vector backend that matches your deployment model.
- Keep the database on durable storage and back it up regularly.
- Use explicit provider configuration rather than implicit defaults in production setups.
