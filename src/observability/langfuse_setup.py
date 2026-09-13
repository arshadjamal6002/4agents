"""
Langfuse observability — Version 6 (cloud or self-hosted).

Attach one CallbackHandler to graph.invoke via get_run_config().
If keys are missing, the app runs without tracing (same as Versions 1–5).

Supports both:
  LANGFUSE_HOST=...          (book / self-hosted)
  LANGFUSE_BASE_URL=...      (Langfuse Cloud UI default)
"""

from __future__ import annotations

import os


def _strip_env(name: str) -> str:
    """Read env var and strip whitespace / surrounding quotes."""
    return os.getenv(name, "").strip().strip('"').strip("'")


def _langfuse_host() -> str:
    return (
        _strip_env("LANGFUSE_HOST")
        or _strip_env("LANGFUSE_BASE_URL")
        or "https://cloud.langfuse.com"
    )


def _langfuse_configured() -> bool:
    return bool(_strip_env("LANGFUSE_PUBLIC_KEY") and _strip_env("LANGFUSE_SECRET_KEY"))


def _ensure_langfuse_env() -> None:
    """
    Normalize env so the Langfuse SDK finds credentials.

    Cloud dashboards often set LANGFUSE_BASE_URL; the SDK also reads
    LANGFUSE_HOST. We mirror whichever is set.
    """
    public = _strip_env("LANGFUSE_PUBLIC_KEY")
    secret = _strip_env("LANGFUSE_SECRET_KEY")
    host = _langfuse_host()

    if public:
        os.environ["LANGFUSE_PUBLIC_KEY"] = public
    if secret:
        os.environ["LANGFUSE_SECRET_KEY"] = secret
    os.environ["LANGFUSE_HOST"] = host
    os.environ["LANGFUSE_BASE_URL"] = host


def get_langfuse_handler(session_id: str, user_id: str = "local"):
    """
    Create a Langfuse LangChain CallbackHandler, or None if not configured.

    Session/user tags are applied via invoke config metadata (see get_run_config),
    because current CallbackHandler only accepts public_key / update_trace.
    """
    if not _langfuse_configured():
        return None

    _ensure_langfuse_env()

    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler

        # Initialize the singleton client from env before the handler attaches.
        get_client()

        return CallbackHandler(
            public_key=_strip_env("LANGFUSE_PUBLIC_KEY") or None,
            update_trace=True,
        )
    except ImportError:
        print(
            "[Observability] langfuse/langchain missing. "
            "Run: pip install langfuse && pip install langchain==1.0.0 --no-deps"
        )
        return None
    except Exception as e:
        print(f"[Observability] Failed to create Langfuse handler: {e}")
        return None


def get_run_config(
    session_id: str,
    user_id: str = "local",
    extra_config: dict | None = None,
) -> dict:
    """
    LangGraph invoke config: checkpoint thread_id + optional Langfuse callbacks.
    """
    config: dict = {
        "configurable": {"thread_id": session_id},
        "run_name": f"learning-accelerator-{session_id}",
        "metadata": {
            "langfuse_session_id": session_id,
            "langfuse_user_id": user_id,
            "langfuse_tags": ["learning-accelerator", "openai"],
            "model": _strip_env("OPENAI_MODEL") or "gpt-4o-mini",
            "framework": "langgraph",
        },
    }

    if extra_config:
        # Shallow merge; nested configurable/metadata get updated carefully
        extra_meta = extra_config.pop("metadata", None)
        extra_cfg = extra_config.pop("configurable", None)
        config.update(extra_config)
        if extra_meta:
            config["metadata"].update(extra_meta)
        if extra_cfg:
            config["configurable"].update(extra_cfg)

    handler = get_langfuse_handler(session_id, user_id)
    if handler:
        config["callbacks"] = [handler]
        print(
            f"[Observability] Tracing session {session_id} -> {_langfuse_host()}"
        )
    else:
        print("[Observability] Langfuse not configured. Running without tracing.")

    return config


# Book / older alias
get_langfuse_config = get_run_config


def flush_langfuse() -> None:
    """Flush pending Langfuse events before process exit (no-op if unused)."""
    if not _langfuse_configured():
        return

    _ensure_langfuse_env()
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        try:
            from langfuse import Langfuse

            Langfuse().flush()
        except Exception:
            pass
