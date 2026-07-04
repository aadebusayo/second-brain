from __future__ import annotations

import argparse
import json
from typing import Sequence

from .store import MemoryStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="secondbrain")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status")
    remember_parser = subparsers.add_parser("remember")
    remember_parser.add_argument("text")
    recall_parser = subparsers.add_parser("recall")
    recall_parser.add_argument("query")
    recall_parser.add_argument("--top-k", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "status":
        store = MemoryStore()
        print(json.dumps(store.status(), indent=2, sort_keys=True))
        return 0

    if args.command == "remember":
        store = MemoryStore()
        node = store.remember(args.text)
        print(json.dumps({"node_id": node.id, "text": node.text}, sort_keys=True))
        return 0

    if args.command == "recall":
        store = MemoryStore()
        results = store.recall_naive(args.query, top_k=args.top_k)
        print(json.dumps([{"node_id": node.id, "text": node.text} for node in results], indent=2, sort_keys=True))
        return 0

    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
