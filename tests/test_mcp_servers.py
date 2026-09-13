"""MCP server unit tests — no LLM required (Version 3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_servers import filesystem_server as fs
from mcp_servers.memory_server import (
    clear_store,
    memory_delete,
    memory_get,
    memory_list_keys,
    memory_set,
)


@pytest.fixture(autouse=True)
def _reset_memory():
    clear_store()
    yield
    clear_store()


@pytest.fixture
def notes_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "alpha.md").write_text("# Alpha\n\nclosure concept here\n", encoding="utf-8")
    (notes / "beta.md").write_text("# Beta\n\nother topic\n", encoding="utf-8")
    monkeypatch.setattr(fs, "NOTES_BASE", notes)
    return notes


def test_list_study_files(notes_dir: Path):
    files = fs.list_study_files()
    assert files == ["alpha.md", "beta.md"]


def test_read_study_file(notes_dir: Path):
    content = fs.read_study_file("alpha.md")
    assert "Alpha" in content
    assert "closure" in content


def test_read_missing_file(notes_dir: Path):
    result = fs.read_study_file("missing.md")
    assert result.startswith("Error:")
    assert "Available:" in result


def test_path_traversal_blocked(notes_dir: Path):
    result = fs.read_study_file("../../.env")
    assert "path traversal" in result.lower() or "Error:" in result


def test_search_notes(notes_dir: Path):
    hits = fs.search_notes("closure")
    assert len(hits) >= 1
    assert hits[0]["file"] == "alpha.md"


def test_notes_index_resource(notes_dir: Path):
    index = fs.get_notes_index()
    assert "alpha.md" in index
    assert "beta.md" in index


def test_memory_set_get():
    assert memory_set("s1", "explained_topics", "Closures") == (
        "Stored 'explained_topics' for session 's1'"
    )
    assert memory_get("s1", "explained_topics") == "Closures"
    assert memory_get("s1", "missing") == "null"


def test_memory_list_and_delete():
    memory_set("s1", "a", "1")
    memory_set("s1", "b", "2")
    assert sorted(memory_list_keys("s1")) == ["a", "b"]
    memory_delete("s1", "a")
    assert memory_list_keys("s1") == ["b"]
    assert memory_get("s1", "a") == "null"
