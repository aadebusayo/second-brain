from __future__ import annotations

from typing import Dict, List


def select(activations: Dict[str, float], token_counts: Dict[str, int], budget: int) -> List[str]:
    ranked = sorted(activations.items(), key=lambda item: item[1], reverse=True)
    chosen: List[str] = []
    used = 0
    for node_id, activation in ranked:
        cost = token_counts.get(node_id, 100)
        if used + cost <= budget:
            chosen.append(node_id)
            used += cost
    return chosen
