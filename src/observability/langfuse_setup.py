"""
Run config helpers — Version 5 uses thread_id for checkpoints.

Langfuse arrives in Version 6. Until then this only sets thread_id
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
    # Version 6 will import and configure Langfuse here.
    return None


def get_run_config(session_id: str) -> dict:
    """
    Config passed to every graph.invoke / Command(resume=...) call.

    `configurable.thread_id` MUST stay the same for a session so LangGraph
    reads/writes the same SQLite checkpoint chain.
    """
    config: dict = {
        "configurable": {
            "thread_id": session_id,
        }
    }
    handler = get_langfuse_handler(session_id)
    if handler:
        config["callbacks"] = [handler]
    return config


# Alias matching the book / future Langfuse version
get_langfuse_config = get_run_config


def flush_langfuse() -> None:
    """No-op until Version 6."""
    return None
