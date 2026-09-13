"""Progress Coach stub — ends the Ch2 demo session after approval."""

from __future__ import annotations


def progress_coach_node(state: dict) -> dict:
    """
    Jump current_topic_index past all topics so route_after_coach → END.

    Real coaching + A2A Study Buddy arrive in Chapters 4 and 8.
    """
    roadmap = state.get("roadmap")
    if isinstance(roadmap, dict):
        n = len(roadmap.get("topics", []))
    elif roadmap is not None:
        n = len(roadmap.topics)
    else:
        n = 0

    print(
        "[Progress Coach] Stub (Chapter 4) — ending Chapter 2 session "
        f"(topics planned: {n})."
    )
    return {
        "current_topic_index": n,
        "roadmap": roadmap,
        "error": None,
    }
