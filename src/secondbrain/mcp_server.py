"""
MCP server wrapping secondbrain's public API as MCP tools.

Uses the official `mcp` Python SDK to expose remember, recall, explain,
and consolidate as MCP tools callable from any MCP-compatible agent harness.
"""

from __future__ import annotations

from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .api import SecondBrain


def create_server(brain: SecondBrain | None = None) -> Server:
    """Create and configure an MCP server with the secondbrain tools."""
    if brain is None:
        brain = SecondBrain()

    server = Server("secondbrain")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="remember",
                description="Store a new memory note in the associative graph",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "The text to remember"},
                        "metadata": {"type": "object", "description": "Optional metadata"},
                    },
                    "required": ["text"],
                },
            ),
            types.Tool(
                name="recall",
                description="Retrieve relevant memories using spreading activation",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "token_budget": {"type": "integer", "description": "Max tokens for results"},
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="explain",
                description="Return the activation math that surfaced a node",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "string", "description": "The node ID to explain"},
                    },
                    "required": ["node_id"],
                },
            ),
            types.Tool(
                name="consolidate",
                description="Run a memory consolidation pass",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        if name == "remember":
            node_id = brain.remember(
                arguments.get("text", ""),
                metadata=arguments.get("metadata"),
            )
            return [types.TextContent(type="text", text=str(node_id))]

        if name == "recall":
            nodes = brain.recall(
                arguments.get("query", ""),
                token_budget=arguments.get("token_budget", 2000),
            )
            result = "\n\n---\n\n".join(
                f"[{node.id}] {node.text}" for node in nodes
            )
            return [types.TextContent(type="text", text=result)]

        if name == "explain":
            explanation = brain.explain(arguments.get("node_id", ""))
            import json
            return [types.TextContent(
                type="text",
                text=json.dumps(explanation, indent=2, default=str),
            )]

        if name == "consolidate":
            report = brain.consolidate()
            return [types.TextContent(
                type="text",
                text=f"Consolidation complete: {report.summaries_created} summaries, "
                     f"{report.nodes_demoted} nodes demoted, "
                     f"{report.edges_rewired} edges rewired.",
            )]

        raise ValueError(f"Unknown tool: {name}")

    return server


async def run() -> None:
    """Entry point: start the MCP server over stdio."""
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
