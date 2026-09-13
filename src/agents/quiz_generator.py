"""Quiz Generator stub — full quiz + grading arrives in Version 4."""

from __future__ import annotations


def quiz_generator_node(state: dict) -> dict:
    print("[Quiz Generator] Stub (Version 4) — skipped.")
    return {
        "quiz_results": state.get("quiz_results", []),
        "weak_areas": state.get("weak_areas", []),
        "roadmap": state.get("roadmap"),
        "current_topic_index": state.get("current_topic_index", 0),
        "session_id": state.get("session_id", ""),
        "error": None,
    }
