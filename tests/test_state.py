"""Unit tests for graph state helpers (no LLM)."""

from graph.state import (
    StudyRoadmap,
    Topic,
    get_current_topic,
    initial_state,
    session_is_complete,
)


def test_initial_state_defaults():
    state = initial_state("Learn X", "abc123")
    assert state["goal"] == "Learn X"
    assert state["session_id"] == "abc123"
    assert state["roadmap"] is None
    assert state["approved"] is False
    assert state["current_topic_index"] == 0
    assert state["error"] is None
    assert state["last_explanation"] == ""
    assert state["last_coaching_message"] == ""


def test_get_current_topic_from_dataclass():
    roadmap = StudyRoadmap(
        goal="g",
        total_weeks=1,
        topics=[
            Topic("A", "desc a", 30),
            Topic("B", "desc b", 45),
        ],
    )
    state = initial_state("g", "s1")
    state["roadmap"] = roadmap
    state["current_topic_index"] = 1
    topic = get_current_topic(state)
    assert topic is not None
    assert topic.title == "B"


def test_get_current_topic_from_dict():
    state = initial_state("g", "s1")
    state["roadmap"] = {
        "goal": "g",
        "total_weeks": 1,
        "weekly_hours": 5,
        "topics": [
            {
                "title": "A",
                "description": "d",
                "estimated_minutes": 30,
                "prerequisites": [],
                "status": "pending",
            }
        ],
    }
    topic = get_current_topic(state)
    assert topic is not None
    assert topic.title == "A"


def test_session_is_complete():
    roadmap = StudyRoadmap(
        goal="g",
        total_weeks=1,
        topics=[Topic("A", "d", 30)],
    )
    state = initial_state("g", "s1")
    state["roadmap"] = roadmap
    assert session_is_complete(state) is False
    state["current_topic_index"] = 1
    assert session_is_complete(state) is True


def test_topic_round_trip():
    t = Topic("Closures", "Nested scope", 60, prerequisites=["Functions"])
    restored = Topic.from_dict(t.to_dict())
    assert restored.title == "Closures"
    assert restored.prerequisites == ["Functions"]
