"""
Run config helpers.

Langfuse arrives in Chapter 6. Until then this only sets thread_id
for SQLite checkpointing.
"""

from __future__ import annotations

import os


def get_langfuse_handler(session_id: str):
    """Return a Langfuse callback handler when keys are set; else None."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    if not public_key or not secret_key:
        return None
    # Chapter 6 will import and configure Langfuse here.
    return None


def get_run_config(session_id: str) -> dict:
    """Config passed to every graph.invoke / resume call."""
    config: dict = {
        "configurable": {
            "thread_id": session_id,
        }
    }
    handler = get_langfuse_handler(session_id)
    if handler:
        config["callbacks"] = [handler]
    return config


# Alias matching the book / future Langfuse chapter
get_langfuse_config = get_run_config


def flush_langfuse() -> None:
    """No-op until Chapter 6."""
    return None
