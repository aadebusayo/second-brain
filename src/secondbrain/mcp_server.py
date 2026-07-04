"""
MCP server wrapping secondbrain's public API as MCP tools.

Uses the official `mcp` Python SDK to expose remember, recall, explain,
and consolidate as MCP tools callable from any MCP-compatible agent harness.
"""

from __future__ import annotations

from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server

from .api import SecondBrain


def create_server(brain: SecondBrain | None = None) -> Server:
    """Create and configure an MCP server with the secondbrain tools."""
    if brain is None:
        brain = SecondBrain()

    server = Server("secondbrain")

    @server.tool()
    async def remember(text: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Store a new memory note in the associative graph."""
        node_id = brain.remember(text, metadata=metadata)
        return {"node_id": node_id, "status": "remembered"}

    @server.tool()
    async def recall(query: str, token_budget: int = 2000) -> list[dict[str, Any]]:
        """Retrieve relevant memories using spreading activation."""
        nodes = brain.recall(query, token_budget=token_budget)
        return [
            {"node_id": node.id, "text": node.text, "activation": node.base_activation}
            for node in nodes
        ]

    @server.tool()
    async def explain(node_id: str) -> dict[str, Any]:
        """Return the full activation math that surfaced a node — seed value,
        hop-by-hop contributions, base-level bonus, and final score."""
        return brain.explain(node_id)

    @server.tool()
    async def consolidate() -> dict[str, Any]:
        """Run a consolidation pass: find dense subgraphs, summarise,
        rewire, and demote originals."""
        report = brain.consolidate()
        return {
            "status": "ok",
            "candidates_found": report.candidates_found,
            "summaries_created": report.summaries_created,
            "edges_rewired": report.edges_rewired,
            "nodes_demoted": report.nodes_demoted,
        }

    return server


async def run() -> None:
    """Entry point: start the MCP server over stdio."""
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
