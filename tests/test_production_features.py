import os
import tempfile

import pytest
from sqlmodel import Session

from secondbrain.config import Settings
from secondbrain.models import MemoryNode
from secondbrain.storage import StorageBackend
from secondbrain.store import MemoryStore


def test_settings_reads_environment(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "anthropic")
    monkeypatch.setenv("VECTOR_PROVIDER", "lancedb")
    settings = Settings()
    assert settings.embedding_provider == "anthropic"
    assert settings.vector_provider == "lancedb"


def test_memory_store_persists_nodes_across_instances():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "secondbrain.sqlite")
        os.environ["SECOND_BRAIN_DB_PATH"] = db_path

        first_store = MemoryStore()
        first_store.remember("persisted note")

        second_store = MemoryStore()
        nodes = second_store.graph.list_nodes()
        assert len(nodes) == 1
        assert nodes[0].text == "persisted note"


def test_settings_rejects_unsupported_values(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "unsupported")
    with pytest.raises(ValueError):
        Settings()


def test_memory_store_reports_runtime_status():
    store = MemoryStore()
    status = store.status()
    assert status["node_count"] == 0
    assert status["embedding_provider"] == store.settings.embedding_provider
    assert status["vector_provider"] == store.settings.vector_provider
    assert "backend" in status


def test_storage_backend_recovers_from_invalid_json_payloads(tmp_path, monkeypatch):
    db_path = tmp_path / "secondbrain.sqlite"
    monkeypatch.setenv("SECOND_BRAIN_DB_PATH", str(db_path))
    storage = StorageBackend()
    with Session(storage.engine) as session:
        session.add(MemoryNode(node_id="bad-row", text="broken", embedding_json="{not-json", metadata_json="{not-json"))
        session.commit()

    loaded = storage.load_nodes()
    assert len(loaded) == 1
    assert loaded[0]["embedding"] == []
    assert loaded[0]["metadata"] == {}
    assert loaded[0]["schema_version"] == 1
    assert loaded[0]["provenance"] == {}
