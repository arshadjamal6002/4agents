"""Progress Coach stub — Version 4 will replace this.

For Version 3: after one Explainer pass, jump past all topics so the
graph ends (avoids burning API calls on a full topic loop).
"""

from __future__ import annotations


def progress_coach_node(state: dict) -> dict:
    roadmap = state.get("roadmap")
    if isinstance(roadmap, dict):
        n = len(roadmap.get("topics", []))
    elif roadmap is not None:
        n = len(roadmap.topics)
    else:
        n = 0

    print(
        "[Progress Coach] Stub (Version 4) — ending after Explainer demo "
        f"(topics planned: {n})."
    )
    return {
        "current_topic_index": n,
        "roadmap": roadmap,
        "error": None,
    }
