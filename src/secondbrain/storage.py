from __future__ import annotations

import json
import os
import time
from typing import Optional

from sqlmodel import Session, SQLModel, create_engine, select

from .config import Settings
from .models import MemoryEdge, MemoryNode


class StorageBackend:
    """Database-backed storage for memory nodes and metadata."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        os.makedirs(os.path.dirname(self.settings.db_path) or ".", exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.settings.db_path}")
        SQLModel.metadata.create_all(self.engine)
        self._ensure_schema_columns()

    def _ensure_schema_columns(self) -> None:
        with self.engine.begin() as connection:
            table_info = connection.exec_driver_sql("PRAGMA table_info(memorynode)").fetchall()
            existing_columns = {row[1] for row in table_info}
            if "schema_version" not in existing_columns:
                connection.exec_driver_sql("ALTER TABLE memorynode ADD COLUMN schema_version INTEGER DEFAULT 1")
            if "provenance_json" not in existing_columns:
                connection.exec_driver_sql("ALTER TABLE memorynode ADD COLUMN provenance_json TEXT DEFAULT '{}'" )
            if "updated_at" not in existing_columns:
                connection.exec_driver_sql("ALTER TABLE memorynode ADD COLUMN updated_at INTEGER")

    def save_node(self, node_id: str, text: str, embedding: list[float], metadata: dict) -> None:
        timestamp = int(time.time())
        provenance = {
            "source": metadata.get("source", "unknown"),
            "schema_version": 1,
            "updated_at": timestamp,
        }
        with Session(self.engine) as session:
            statement = select(MemoryNode).where(MemoryNode.node_id == node_id)
            existing = session.exec(statement).first()
            if existing is None:
                record = MemoryNode(
                    node_id=node_id,
                    text=text,
                    embedding_json=json.dumps(embedding),
                    metadata_json=json.dumps(metadata),
                    schema_version=1,
                    provenance_json=json.dumps(provenance),
                    updated_at=timestamp,
                )
                session.add(record)
            else:
                existing.text = text
                existing.embedding_json = json.dumps(embedding)
                existing.metadata_json = json.dumps(metadata)
                existing.schema_version = 1
                existing.provenance_json = json.dumps(provenance)
                existing.updated_at = timestamp
            session.commit()

    def load_nodes(self) -> list[dict]:
        with Session(self.engine) as session:
            rows = session.exec(select(MemoryNode)).all()
            records: list[dict] = []
            for row in rows:
                try:
                    embedding = json.loads(row.embedding_json)
                except (TypeError, ValueError):
                    embedding = []
                try:
                    metadata = json.loads(row.metadata_json)
                except (TypeError, ValueError):
                    metadata = {}
                provenance = {}
                try:
                    provenance = json.loads(row.provenance_json)
                except (TypeError, ValueError):
                    provenance = {}
                records.append(
                    {
                        "node_id": row.node_id,
                        "text": row.text,
                        "embedding": embedding if isinstance(embedding, list) else [],
                        "metadata": metadata if isinstance(metadata, dict) else {},
                        "provenance": provenance if isinstance(provenance, dict) else {},
                        "schema_version": row.schema_version,
                        "updated_at": row.updated_at,
                    }
                )
            return records

    # ------------------------------------------------------------------
    # Edge persistence
    # ------------------------------------------------------------------

    def save_edge(self, source_id: str, target_id: str, weight: float, relation_type: str = "") -> None:
        """Persist a single edge (upsert)."""
        timestamp = int(time.time())
        with Session(self.engine) as session:
            # Delete existing edge between these nodes (any direction)
            stmt = select(MemoryEdge).where(
                (MemoryEdge.source_id == source_id) & (MemoryEdge.target_id == target_id)
            )
            existing = session.exec(stmt).first()
            if existing is None:
                stmt = select(MemoryEdge).where(
                    (MemoryEdge.source_id == target_id) & (MemoryEdge.target_id == source_id)
                )
                existing = session.exec(stmt).first()

            if existing is not None:
                existing.weight = weight
                existing.relation_type = relation_type
                existing.updated_at = timestamp
            else:
                record = MemoryEdge(
                    source_id=source_id,
                    target_id=target_id,
                    weight=weight,
                    relation_type=relation_type,
                    updated_at=timestamp,
                )
                session.add(record)
            session.commit()

    def remove_edge(self, source_id: str, target_id: str) -> None:
        """Remove a persisted edge."""
        with Session(self.engine) as session:
            for stmt in [
                select(MemoryEdge).where(
                    (MemoryEdge.source_id == source_id) & (MemoryEdge.target_id == target_id)
                ),
                select(MemoryEdge).where(
                    (MemoryEdge.source_id == target_id) & (MemoryEdge.target_id == source_id)
                ),
            ]:
                existing = session.exec(stmt).first()
                if existing is not None:
                    session.delete(existing)
            session.commit()

    def load_edges(self) -> list[dict]:
        """Load all persisted edges."""
        with Session(self.engine) as session:
            rows = session.exec(select(MemoryEdge)).all()
            return [
                {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "weight": r.weight,
                    "relation_type": r.relation_type,
                }
                for r in rows
            ]
