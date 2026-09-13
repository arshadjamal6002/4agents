"""Unit tests for Quiz Generator and Progress Coach (Version 4, no live API)."""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage

from agents.progress_coach import PASS_THRESHOLD, progress_coach_node
from agents.quiz_generator import (
    extract_explanation,
    parse_grade_payload,
    quiz_generator_node,
)
from graph.state import QuizResult, StudyRoadmap, Topic, initial_state


def test_extract_explanation_skips_tool_calls():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "tool_list_files", "args": {}, "id": "1"}]),
        AIMessage(content="Here is the real explanation."),
    ]
    assert extract_explanation(messages) == "Here is the real explanation."


def test_extract_explanation_skips_memory_ack():
    real = (
        "Think of an if-statement like a fork in the road.\n\n"
        "Core idea: conditionals choose a branch.\n\n"
        "```python\nif n > 10:\n    print('big')\n```\n\n"
        "Common mistake: using = instead of ==."
    )
    messages = [
        AIMessage(content=real),
        AIMessage(
            content=(
                "I've stored the explanation about Control Structures for future "
                "reference. If you have any more questions, feel free to ask!"
            )
        ),
    ]
    assert extract_explanation(messages) == real


def test_parse_grade_payload_clamps_score():
    parsed = parse_grade_payload(
        {"correct": True, "score": 1.5, "feedback": "Great", "missing_concept": ""}
    )
    assert parsed["score"] == 1.0
    assert parsed["correct"] is True


def test_quiz_generator_accumulates_results():
    state = initial_state("Learn X", "sess1")
    state["roadmap"] = StudyRoadmap(
        goal="Learn X",
        total_weeks=1,
        topics=[Topic("Closures", "Nested functions", 60)],
    )
    state["messages"] = [AIMessage(content="Closures remember enclosing vars.")]

    fake_result = QuizResult(
        topic="Closures",
        questions=[],
        score=0.8,
        weak_areas=["nonlocal"],
    )

    with patch("agents.quiz_generator.run_quiz", return_value=fake_result):
        out = quiz_generator_node(state)

    assert len(out["quiz_results"]) == 1
    assert out["quiz_results"][0].score == 0.8
    assert "nonlocal" in out["weak_areas"]
    assert out["error"] is None


def test_progress_coach_marks_completed_and_advances():
    state = initial_state("Learn X", "sess1")
    roadmap = StudyRoadmap(
        goal="Learn X",
        total_weeks=1,
        topics=[
            Topic("A", "desc", 30),
            Topic("B", "desc", 30),
        ],
    )
    state["roadmap"] = roadmap
    state["current_topic_index"] = 0
    state["quiz_results"] = [
        QuizResult(topic="A", questions=[], score=0.9, weak_areas=[]),
    ]

    with patch(
        "agents.progress_coach.get_coaching_message",
        return_value={"summary": "Nice work on A.", "encouragement": "Onward!"},
    ), patch("agents.progress_coach.memory_set") as mem:
        out = progress_coach_node(state)

    assert out["current_topic_index"] == 1
    assert out["roadmap"].topics[0].status == "completed"
    assert out["error"] is None
    mem.assert_called_once()


def test_progress_coach_marks_needs_review_on_fail():
    state = initial_state("Learn X", "sess1")
    state["roadmap"] = StudyRoadmap(
        goal="g",
        total_weeks=1,
        topics=[Topic("A", "d", 30)],
    )
    state["quiz_results"] = [
        QuizResult(topic="A", questions=[], score=PASS_THRESHOLD - 0.1, weak_areas=["scope"]),
    ]

    with patch(
        "agents.progress_coach.get_coaching_message",
        return_value={"summary": "More practice.", "encouragement": "Retry soon."},
    ), patch("agents.progress_coach.memory_set"):
        out = progress_coach_node(state)

    assert out["roadmap"].topics[0].status == "needs_review"
    assert out["current_topic_index"] == 1
