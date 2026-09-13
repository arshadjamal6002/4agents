"""
Learning Accelerator entry point.

Usage:
  python main.py "Learn Python closures from scratch"
  python main.py --resume <session-id>
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from langgraph.types import Command

from graph.state import StudyRoadmap, initial_state
from graph.workflow import graph
from observability.langfuse_setup import flush_langfuse, get_run_config


def run_session(goal: str, session_id: str | None = None) -> None:
    is_resume = session_id is not None
    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    config = get_run_config(session_id)

    print(f"\n{'=' * 60}")
    print("Learning Accelerator")
    print(f"Session ID: {session_id}")
    if is_resume:
        print("Resuming existing session...")
    else:
        print(f"Goal: {goal}")
    print(f"{'=' * 60}")

    state = None if is_resume else initial_state(goal, session_id)

    try:
        result = graph.invoke(state, config=config)
    except Exception as e:
        if is_resume:
            print(f"\n[ERROR] Could not resume session '{session_id}': {e}")
            print(
                "Check the session ID or that data/checkpoints.db still exists."
            )
            return
        raise

    while "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value
        raw_roadmap = interrupt_payload.get("roadmap")
        roadmap = (
            StudyRoadmap.from_dict(raw_roadmap)
            if isinstance(raw_roadmap, dict)
            else raw_roadmap
        )

        if roadmap:
            print(f"\n{'=' * 60}")
            print("Proposed Study Plan")
            print(f"{'=' * 60}")
            print(f"Goal: {roadmap.goal}")
            print(
                f"Duration: {roadmap.total_weeks} weeks @ "
                f"{roadmap.weekly_hours} hrs/week\n"
            )
            for i, topic in enumerate(roadmap.topics, 1):
                prereqs = (
                    f" (needs: {', '.join(topic.prerequisites)})"
                    if topic.prerequisites
                    else ""
                )
                print(
                    f"  {i}. {topic.title} "
                    f"({topic.estimated_minutes} min){prereqs}"
                )
                print(f"     {topic.description}")

        print(f"\n{interrupt_payload.get('prompt', 'Continue?')}")
        user_input = input("> ").strip()
        result = graph.invoke(Command(resume=user_input), config=config)

    if result.get("error"):
        print(f"\n[ERROR] {result['error']}")
        flush_langfuse()
        return

    print(f"\n{'=' * 60}")
    print("Version 3 complete (Planner + MCP Explainer).")
    print("Quiz / Coach are stubs until Version 4.")
    print(f"Save this Session ID to resume later: {session_id}")
    print(f"{'=' * 60}\n")
    flush_langfuse()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Learning Accelerator: multi-agent study system "
            "(OpenAI + LangGraph + MCP). Versions 1-3 implemented."
        ),
        epilog=(
            "Examples:\n"
            '  python main.py "Learn Python closures from scratch"\n'
            "  python main.py --resume a3f1b2c4\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "goal",
        nargs="?",
        default="Learn Python closures and decorators from scratch",
        help="What you want to learn",
    )
    parser.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="Resume an existing session by its 8-char ID",
    )
    args = parser.parse_args()

    if args.resume:
        run_session(goal="", session_id=args.resume)
    else:
        run_session(goal=args.goal)
