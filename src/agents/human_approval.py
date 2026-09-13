"""Human-in-the-loop roadmap approval — Version 5 (interrupt / resume)."""

from __future__ import annotations

from langgraph.types import interrupt

from graph.checkpointing import coerce_roadmap


def human_approval_node(state: dict) -> dict:
    """
    Pause for roadmap approval via interrupt().

    interrupt() checkpoints state and returns control to the caller.
    The caller resumes with Command(resume=user_input). Execution continues
    on the next line with `decision` set to that value.

    LangGraph 1.1 caveat: after resume, downstream nodes may only see keys
    this node returns — so we re-emit every field later agents need.
    """
    roadmap = coerce_roadmap(state.get("roadmap"))

    if roadmap is None:
        return {"approved": True}

    print("\n[Human Approval] Pausing for roadmap review...")

    # Pass a plain dict in the payload so interrupt/checkpoint round-trips
    # stay JSON-friendly; main.py coerces back to StudyRoadmap.
    decision = interrupt(
        {
            "type": "roadmap_approval",
            "roadmap": roadmap.to_dict(),
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

    # Re-emit full contract after interrupt/resume (Version 5 requirement).
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
