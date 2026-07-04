"""Tests for the MCP server wrapping secondbrain's public API."""
import tempfile
from secondbrain.api import SecondBrain
from secondbrain.config import Settings
from secondbrain.entity import EntityModel
from secondbrain.graph import MemoryGraph
from secondbrain.store import MemoryStore


def _clean_brain():
    db_path = tempfile.mktemp(suffix=".sqlite")
    settings = Settings()
    object.__setattr__(settings, "db_path", db_path)
    object.__setattr__(settings, "wire_threshold", 0.4)
    object.__setattr__(settings, "embedding_provider", "local")
    graph = MemoryGraph()
    store = MemoryStore(graph=graph, settings=settings)
    brain = SecondBrain.__new__(SecondBrain)
    brain.graph = graph
    brain.store = store
    brain.entities = EntityModel(graph=graph, embedding_provider=store.embedding_provider)
    brain._llm_client = None
    brain._nodes_since_consolidation = 0
    brain._consolidate_every_n = 999999
    return brain


def test_mcp_server_initializes():
    """Server should initialize and report its name."""
    from secondbrain.mcp_server import create_server
    brain = _clean_brain()
    server = create_server(brain=brain)
    assert server.name == "secondbrain"


def test_mcp_tools_end_to_end():
    """The API methods behind the MCP tools work."""
    brain = _clean_brain()
    nid = brain.remember("MCP test: fintech compliance framework")
    results = brain.recall("fintech compliance", token_budget=500)
    assert any(n.id == nid for n in results)
    explanation = brain.explain(nid)
    assert "seed_value" in explanation
    assert "final_score" in explanation
    report = brain.consolidate()
    assert report.candidates_found >= 0
