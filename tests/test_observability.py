"""Observability unit tests — Version 6 (no live Langfuse required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from observability.langfuse_setup import (
    _langfuse_configured,
    flush_langfuse,
    get_langfuse_handler,
    get_run_config,
)


def test_not_configured_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    assert _langfuse_configured() is False
    assert get_langfuse_handler("s1") is None
    config = get_run_config("s1")
    assert config["configurable"]["thread_id"] == "s1"
    assert "callbacks" not in config


def test_configured_with_keys_attaches_callback(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    fake_handler = MagicMock(name="CallbackHandler")
    with (
        patch("langfuse.get_client", return_value=MagicMock()),
        patch(
            "langfuse.langchain.CallbackHandler",
            return_value=fake_handler,
        ) as ctor,
    ):
        handler = get_langfuse_handler("sess-abc")
        assert handler is fake_handler
        ctor.assert_called_once()

        config = get_run_config("sess-abc")
        assert config["callbacks"] == [fake_handler]
        assert config["metadata"]["langfuse_session_id"] == "sess-abc"
        assert config["configurable"]["thread_id"] == "sess-abc"


def test_strips_quotes_from_env(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", '"pk-quoted"')
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "'sk-quoted'")
    assert _langfuse_configured() is True


def test_flush_noop_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    flush_langfuse()  # must not raise


def test_base_url_alias_sets_host(monkeypatch):
    import os

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    with (
        patch("langfuse.get_client", return_value=MagicMock()),
        patch("langfuse.langchain.CallbackHandler", return_value=MagicMock()),
    ):
        get_langfuse_handler("s")

    assert os.environ.get("LANGFUSE_HOST") == "https://cloud.langfuse.com"
