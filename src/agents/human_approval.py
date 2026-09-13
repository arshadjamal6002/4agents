"""Human-in-the-loop roadmap approval (Chapter 2 / detailed in Chapter 5)."""

from __future__ import annotations

from langgraph.types import interrupt


def human_approval_node(state: dict) -> dict:
    """
    Pause for roadmap approval via interrupt().

    After Command(resume=...), LangGraph 1.1 may only forward keys this node
    returns — so we re-emit the fields downstream nodes need.
    """
    roadmap = state.get("roadmap")

    if roadmap is None:
        return {"approved": True}

    print("\n[Human Approval] Pausing for roadmap review...")

    decision = interrupt(
        {
            "type": "roadmap_approval",
            "roadmap": roadmap,
            "prompt": (
                "Does this study plan look good?\n"
                "  Type 'yes' to start studying\n"
                "  Type 'no' to generate a different plan"
            ),
        }
    )

    approved = str(decision).lower().strip() in ("yes", "y", "ok", "approve")

    if approved:
        print("[Human Approval] Roadmap approved. Starting study session.")
    else:
        print("[Human Approval] Roadmap rejected. Regenerating...")

    return {
        "approved": approved,
        "roadmap": roadmap,
        "goal": state.get("goal", ""),
        "session_id": state.get("session_id", ""),
        "current_topic_index": state.get("current_topic_index", 0),
        "quiz_results": state.get("quiz_results", []),
        "weak_areas": state.get("weak_areas", []),
        "study_materials_path": state.get(
            "study_materials_path", "study_materials/sample_notes"
        ),
        "error": None,
    }
