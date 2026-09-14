"""
Shared fixtures for the Learning Accelerator test suite — Version 7.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_roadmap():
    """A minimal StudyRoadmap for use in unit tests."""
    from graph.state import StudyRoadmap, Topic

    return StudyRoadmap(
        goal="Learn Python closures",
        total_weeks=2,
        topics=[
            Topic(
                title="Closures Explained",
                description="Understand how closures capture enclosing scope variables",
                estimated_minutes=60,
            ),
            Topic(
                title="Practical Closure Patterns",
                description="Apply closures to real problems: factories, memoisation",
                estimated_minutes=45,
                prerequisites=["Closures Explained"],
            ),
        ],
    )


@pytest.fixture
def sample_state(sample_roadmap):
    """A minimal AgentState dict for use in unit tests."""
    from graph.state import initial_state

    state = initial_state("Learn Python closures", "test-session-001")
    state["roadmap"] = sample_roadmap
    state["current_topic_index"] = 0
    return state


@pytest.fixture
def closures_note_content():
    """
    The content of closures.md, used as retrieval context in faithfulness tests.
    Falls back to an inline summary if the file doesn't exist.
    """
    notes_path = (
        Path(__file__).parent.parent
        / "study_materials/sample_notes/closures.md"
    )
    if notes_path.exists():
        return notes_path.read_text(encoding="utf-8")
    return (
        "A closure is a nested function that remembers variables from its "
        "enclosing scope even after the enclosing function returns."
    )
