"""Explainer stub — full MCP tool loop arrives in Chapter 3."""

from __future__ import annotations


def explainer_node(state: dict) -> dict:
    topic_index = state.get("current_topic_index", 0)
    print(
        f"\n[Explainer] Stub (Chapter 3) — topic index {topic_index}. "
        "MCP + tool-calling loop not implemented yet."
    )
    return {"error": None}
