"""Transparent MIRA-style memory importance scoring.

This is a decision-support utility, not an embedding model or validated claim
about a fact's true importance. Relevance/usefulness/confidence signals must be
provided by the caller and normalized to [0, 1]. Redundancy is a [0, 1]
penalty, where 1 means fully redundant.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any

WEIGHTS = {
    "semantic_relevance": 0.25,
    "historical_usefulness": 0.20,
    "query_context_usefulness": 0.25,
    "non_redundancy": 0.15,
    "confidence": 0.15,
}
REQUIRED = ("semantic_relevance", "historical_usefulness",
            "query_context_usefulness", "redundancy", "confidence")


def score_memory_items(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Rank memory records by transparent, weighted importance signals.

    Each item must have a unique non-empty ``id`` and numeric values in [0, 1]
    for semantic_relevance, historical_usefulness, query_context_usefulness,
    redundancy, and confidence. Higher scores indicate higher estimated
    importance/utility, not a recommendation to retain sensitive information.
    """
    if not items:
        raise ValueError("items must be a non-empty sequence")
    seen: set[str] = set()
    scored = []
    for index, item in enumerate(items):
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError(f"item {index} requires a non-empty string id")
        if item_id in seen:
            raise ValueError(f"duplicate item id: {item_id}")
        seen.add(item_id)
        signals: dict[str, float] = {}
        for key in REQUIRED:
            value = item.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{item_id}.{key} must be numeric in [0, 1]")
            value = float(value)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{item_id}.{key} must be in [0, 1]")
            signals[key] = value
        components = {
            "semantic_relevance": signals["semantic_relevance"],
            "historical_usefulness": signals["historical_usefulness"],
            "query_context_usefulness": signals["query_context_usefulness"],
            "non_redundancy": 1.0 - signals["redundancy"],
            "confidence": signals["confidence"],
        }
        score = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
        scored.append({"id": item_id, "score": round(score, 6),
                       "components": components, "signals": signals})
    scored.sort(key=lambda row: (-row["score"], row["id"]))
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank
    return {
        "method": "MIRA-prototype-v1",
        "weights": dict(WEIGHTS),
        "items": scored,
        "limitations": [
            "Input signals are caller-supplied and are not inferred or independently validated by this utility.",
            "Weights are transparent prototype assumptions, not empirically calibrated parameters.",
            "Importance ranking is decision support and does not establish successful unlearning or justify retaining personal data.",
        ],
    }
