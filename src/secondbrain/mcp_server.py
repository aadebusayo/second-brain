from __future__ import annotations

from typing import Any, Dict

from .api import SecondBrain


class MCPServer:
    def __init__(self) -> None:
        self.brain = SecondBrain()

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": "remember", "description": "Store a note"},
            {"name": "recall", "description": "Retrieve relevant memories"},
            {"name": "explain", "description": "Explain retrieval rationale"},
            {"name": "consolidate", "description": "Run consolidation"},
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name == "remember":
            return self.brain.remember(arguments.get("text", ""), metadata=arguments.get("metadata"))
        if name == "recall":
            return [node.text for node in self.brain.recall(arguments.get("query", ""))]
        if name == "explain":
            return self.brain.explain(arguments.get("node_id", ""))
        if name == "consolidate":
            return self.brain.consolidate()
        raise ValueError(f"unknown tool: {name}")
