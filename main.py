"""
Learning Accelerator entry point.

Usage:
  python main.py "Learn Python closures from scratch"
  python main.py --resume <session-id>
  python main.py --list-sessions
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from langgraph.types import Command

from graph.checkpointing import (
    checkpoint_db_path,
    coerce_roadmap,
    list_session_ids,
    session_exists,
)
from graph.state import QuizResult, initial_state
from graph.workflow import graph
from observability.langfuse_setup import flush_langfuse, get_run_config


def print_session_summary(result: dict) -> None:
    """Print quiz scores after a completed session."""
    roadmap = coerce_roadmap(result.get("roadmap"))
    if roadmap is None:
        return

    raw_results = result.get("quiz_results", [])
    quiz_results = [
        QuizResult.from_dict(r) if isinstance(r, dict) else r for r in raw_results
    ]
    if not quiz_results:
        return

    print(f"\n{'=' * 60}")
    print("Session Summary")
    print(f"{'=' * 60}")
    print(f"Goal: {roadmap.goal}")
    print(f"Topics covered: {len(quiz_results)}/{len(roadmap.topics)}")

    avg = sum(r.score for r in quiz_results) / len(quiz_results)
    print(f"Average score: {avg:.0%}\n")

    for r in quiz_results:
        status = "OK" if r.score >= 0.5 else "X"
        weak = f", review: {', '.join(r.weak_areas)}" if r.weak_areas else ""
        print(f"  {status} {r.topic}: {r.score:.0%}{weak}")

    all_weak = result.get("weak_areas", [])
    if all_weak:
        print(f"\nTopics to revisit: {', '.join(all_weak)}")
    print(f"{'=' * 60}\n")


def print_interrupt_roadmap(interrupt_payload: dict) -> None:
    roadmap = coerce_roadmap(interrupt_payload.get("roadmap"))
    if not roadmap:
        return

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


def run_session(goal: str, session_id: str | None = None) -> None:
    is_resume = session_id is not None
    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    if is_resume and not session_exists(session_id):
        print(f"\n[ERROR] No checkpoint found for session '{session_id}'.")
        print(f"Database: {checkpoint_db_path()}")
        known = list_session_ids()
        if known:
            print("Known sessions:")
            for sid in known:
                print(f"  - {sid}")
        else:
            print("No sessions in the checkpoint database yet.")
        return

    # Same config (thread_id) on every invoke/resume — that is how LangGraph
    # loads the right checkpoint row (Version 5).
    config = get_run_config(session_id)

    print(f"\n{'=' * 60}")
    print("Learning Accelerator")
    print(f"Session ID: {session_id}")
    if is_resume:
        print("Resuming existing session from SQLite checkpoint...")
    else:
        print(f"Goal: {goal}")
        print("Tip: if you stop mid-run, resume with:")
        print(f'  python main.py --resume {session_id}')
    print(f"{'=' * 60}")

    # New session: provide initial state. Resume: pass None so LangGraph
    # loads the latest checkpoint for this thread_id.
    state = None if is_resume else initial_state(goal, session_id)

    try:
        result = graph.invoke(state, config=config)

        # HITL: interrupt() inside human_approval_node returns "__interrupt__".
        # Rejecting the plan re-runs the planner and may interrupt again.
        while "__interrupt__" in result:
            interrupt_payload = result["__interrupt__"][0].value
            print_interrupt_roadmap(interrupt_payload)
            print(f"\n{interrupt_payload.get('prompt', 'Continue?')}")
            user_input = input("> ").strip()
            result = graph.invoke(Command(resume=user_input), config=config)

        if result.get("error"):
            print(f"\n[ERROR] {result['error']}")
            return

        print_session_summary(result)
        print(f"\n{'=' * 60}")
        print(
            "Session finished (Versions 1-6: plan, explain, quiz, "
            "coach, checkpoints, traces)."
        )
        print(f"Session ID: {session_id}")
        print(f"{'=' * 60}\n")
    except KeyboardInterrupt:
        print("\n[Interrupted] Session paused. Resume later with:")
        print(f"  python main.py --resume {session_id}")
    except Exception as e:
        if is_resume:
            print(f"\n[ERROR] Could not resume session '{session_id}': {e}")
            print(
                "Check the session ID or that data/checkpoints.db still exists."
            )
            return
        raise
    finally:
        # Always flush traces (Ctrl+C used to skip this and empty Langfuse).
        flush_langfuse()


def print_sessions() -> None:
    db = checkpoint_db_path()
    sessions = list_session_ids(db)
    print(f"Checkpoint DB: {db}")
    if not sessions:
        print("No saved sessions.")
        return
    print(f"Saved sessions ({len(sessions)}):")
    for sid in sessions:
        print(f"  - {sid}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Learning Accelerator: multi-agent study system "
            "(OpenAI + LangGraph + MCP + Langfuse). Versions 1-6 implemented."
        ),
        epilog=(
            "Examples:\n"
            '  python main.py "Learn Python closures from scratch"\n'
            "  python main.py --resume a3f1b2c4\n"
            "  python main.py --list-sessions\n"
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
        help="Resume an existing session by its ID (loads SQLite checkpoint)",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List thread_ids stored in the checkpoint database",
    )
    args = parser.parse_args()

    if args.list_sessions:
        print_sessions()
    elif args.resume:
        run_session(goal="", session_id=args.resume)
    else:
        run_session(goal=args.goal)
