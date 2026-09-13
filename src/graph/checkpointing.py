"""
Checkpoint / session helpers — Version 5.

LangGraph persists AgentState to SQLite after every node under thread_id
(= session_id). These helpers make resume and debugging safer.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from graph.state import StudyRoadmap


def checkpoint_db_path() -> str:
    return os.getenv("CHECKPOINT_DB", "data/checkpoints.db")


def ensure_data_dir() -> Path:
    path = Path("data")
    path.mkdir(exist_ok=True)
    return path


def coerce_roadmap(raw) -> StudyRoadmap | None:
    """Normalize roadmap from live state or checkpoint/interrupt payload."""
    if raw is None:
        return None
    if isinstance(raw, StudyRoadmap):
        return raw
    if isinstance(raw, dict):
        return StudyRoadmap.from_dict(raw)
    return None


def list_session_ids(db_path: str | None = None) -> list[str]:
    """
    Return distinct thread_ids stored in the checkpoint DB.

    Empty list if the DB file does not exist yet.
    """
    path = db_path or checkpoint_db_path()
    if not Path(path).exists():
        return []

    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    return [r[0] for r in rows if r and r[0]]


def session_exists(session_id: str, db_path: str | None = None) -> bool:
    """True if any checkpoint exists for this session / thread_id."""
    return session_id in list_session_ids(db_path)


def summarize_checkpoint_values(values: dict | None) -> dict:
    """Build a small UI-friendly summary from graph state values."""
    values = values or {}
    roadmap = coerce_roadmap(values.get("roadmap"))
    quiz_results = values.get("quiz_results") or []
    scores = []
    for r in quiz_results:
        if isinstance(r, dict):
            scores.append(float(r.get("score", 0.0)))
        else:
            scores.append(float(getattr(r, "score", 0.0)))
    avg = sum(scores) / len(scores) if scores else None
    idx = int(values.get("current_topic_index") or 0)
    total = len(roadmap.topics) if roadmap else 0
    complete = total > 0 and idx >= total
    return {
        "session_id": values.get("session_id") or "",
        "goal": values.get("goal") or (roadmap.goal if roadmap else ""),
        "roadmap": roadmap,
        "quiz_results": quiz_results,
        "weak_areas": values.get("weak_areas") or [],
        "current_topic_index": idx,
        "topic_count": total,
        "average_score": avg,
        "complete": complete,
        "last_explanation": values.get("last_explanation") or "",
        "last_coaching_message": values.get("last_coaching_message") or "",
    }
