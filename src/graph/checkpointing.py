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
