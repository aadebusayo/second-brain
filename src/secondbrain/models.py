from __future__ import annotations

from typing import Optional

from sqlmodel import Field, SQLModel


class MemoryNode(SQLModel, table=True):
    """SQL-backed persistence model for a memory node."""

    id: Optional[int] = Field(default=None, primary_key=True)
    node_id: str = Field(index=True, unique=True)
    text: str
    embedding_json: str = "[]"
    metadata_json: str = "{}"
    schema_version: int = Field(default=1)
    provenance_json: str = "{}"
    updated_at: Optional[int] = Field(default=None)
