"""
Version 5 — checkpointing and human-in-the-loop tests.

Covers:
  - SqliteSaver / build_graph DB creation
  - human_approval_node yes/no (interrupt mocked)
  - routing + session_is_complete edge cases
  - coerce_roadmap / list_session_ids helpers
  - real interrupt → resume on a compiled graph (planner mocked)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

from agents.human_approval import human_approval_node
from graph.checkpointing import (
    coerce_roadmap,
    list_session_ids,
    session_exists,
)
from graph.state import (
    QuizResult,
    StudyRoadmap,
    Topic,
    get_current_topic,
    initial_state,
    session_is_complete,
)
from graph.workflow import build_graph, route_after_approval, route_after_coach


def _sample_roadmap() -> StudyRoadmap:
    return StudyRoadmap(
        goal="Learn Python",
        total_weeks=2,
        weekly_hours=5,
        topics=[
            Topic("Closures", "Understanding closures", 60),
            Topic(
                "Decorators",
                "Building decorators",
                75,
                prerequisites=["Closures"],
            ),
        ],
    )


def _close_graph(g) -> None:
    """Close the SqliteSaver connection so Windows can delete temp DB files."""
    conn = getattr(g, "_checkpoint_conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


# ── SqliteSaver / helpers ───────────────────────────────────────────────────


class TestSqliteAndHelpers:
    def test_sqlite_saver_importable(self):
        from langgraph.checkpoint.sqlite import SqliteSaver

        assert SqliteSaver is not None

    def test_build_graph_creates_db_file(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = os.path.join(tmpdir, "checkpoints.db")
            g = build_graph(db_path=db_path)
            try:
                assert g is not None
                assert os.path.exists(db_path)
            finally:
                _close_graph(g)

    def test_coerce_roadmap_from_dict_and_dataclass(self):
        roadmap = _sample_roadmap()
        assert coerce_roadmap(roadmap) is roadmap
        restored = coerce_roadmap(roadmap.to_dict())
        assert restored is not None
        assert restored.goal == "Learn Python"
        assert len(restored.topics) == 2
        assert coerce_roadmap(None) is None

    def test_list_session_ids_empty_missing_db(self):
        assert list_session_ids("definitely_missing_checkpoints.db") == []

    def test_get_current_topic_after_dict_deserialization(self):
        state = initial_state("g", "s1")
        state["roadmap"] = _sample_roadmap().to_dict()
        state["current_topic_index"] = 1
        topic = get_current_topic(state)
        assert topic is not None
        assert topic.title == "Decorators"


# ── Human approval ──────────────────────────────────────────────────────────


class TestHumanApprovalNode:
    def _state(self):
        state = initial_state("Learn Python", "session-test")
        state["roadmap"] = _sample_roadmap()
        return state

    @patch("agents.human_approval.interrupt")
    def test_yes_sets_approved(self, mock_interrupt):
        mock_interrupt.return_value = "yes"
        result = human_approval_node(self._state())
        assert result["approved"] is True
        assert result["session_id"] == "session-test"
        assert result["roadmap"] is not None
        payload = mock_interrupt.call_args[0][0]
        assert isinstance(payload["roadmap"], dict)

    @patch("agents.human_approval.interrupt")
    def test_no_sets_not_approved(self, mock_interrupt):
        mock_interrupt.return_value = "no"
        assert human_approval_node(self._state())["approved"] is False

    @pytest.mark.parametrize("answer", ["y", "ok", "YES", " approve "])
    @patch("agents.human_approval.interrupt")
    def test_approval_aliases(self, mock_interrupt, answer):
        mock_interrupt.return_value = answer
        assert human_approval_node(self._state())["approved"] is True

    def test_no_roadmap_auto_approves(self):
        state = initial_state("test", "s1")
        state["roadmap"] = None
        assert human_approval_node(state)["approved"] is True


# ── Routing ─────────────────────────────────────────────────────────────────


class TestRouting:
    def test_approved_to_explainer(self):
        state = initial_state("t", "s")
        state["approved"] = True
        assert route_after_approval(state) == "explainer"

    def test_rejected_to_planner(self):
        state = initial_state("t", "s")
        state["approved"] = False
        assert route_after_approval(state) == "curriculum_planner"

    def test_coach_loop_and_end(self):
        state = initial_state("t", "s")
        state["roadmap"] = _sample_roadmap()
        state["current_topic_index"] = 0
        assert route_after_coach(state) == "explainer"
        state["current_topic_index"] = 2
        assert route_after_coach(state) == "end"

    def test_session_complete_edge_cases(self):
        state = initial_state("t", "s")
        assert session_is_complete(state) is True
        state["roadmap"] = _sample_roadmap()
        state["current_topic_index"] = 0
        assert session_is_complete(state) is False
        state["current_topic_index"] = 99
        assert session_is_complete(state) is True


# ── Interrupt / resume on compiled graph ────────────────────────────────────


def _fake_planner(state: dict) -> dict:
    return {
        "roadmap": _sample_roadmap(),
        "messages": [],
        "error": None,
    }


def _fake_explainer(state: dict) -> dict:
    return {
        "messages": [AIMessage(content="Explanation of closures.")],
        "error": None,
    }


def _fake_quiz(state: dict) -> dict:
    return {
        "quiz_results": [
            QuizResult(
                topic="Closures",
                questions=[],
                score=0.8,
                weak_areas=[],
            )
        ],
        "weak_areas": [],
        "roadmap": state.get("roadmap"),
        "current_topic_index": state.get("current_topic_index", 0),
        "session_id": state.get("session_id", ""),
        "error": None,
    }


def _fake_coach_end(state: dict) -> dict:
    roadmap = coerce_roadmap(state.get("roadmap"))
    n = len(roadmap.topics) if roadmap else 0
    return {
        "roadmap": roadmap,
        "current_topic_index": n,
        "messages": [AIMessage(content="Done.")],
        "error": None,
    }


class TestInterruptResumeGraph:
    def test_interrupt_then_approve_then_finish(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = str(Path(tmpdir) / "cp.db")
            with (
                patch("graph.workflow.curriculum_planner_node", _fake_planner),
                patch("graph.workflow.explainer_node", _fake_explainer),
                patch("graph.workflow.quiz_generator_node", _fake_quiz),
                patch("graph.workflow.progress_coach_node", _fake_coach_end),
            ):
                g = build_graph(db_path=db_path)
                try:
                    session_id = "v5-test-approve"
                    config = {"configurable": {"thread_id": session_id}}

                    result = g.invoke(
                        initial_state("Learn Python", session_id),
                        config=config,
                    )
                    assert "__interrupt__" in result
                    payload = result["__interrupt__"][0].value
                    assert payload["type"] == "roadmap_approval"
                    assert isinstance(payload["roadmap"], dict)

                    assert session_exists(session_id, db_path)

                    result = g.invoke(Command(resume="yes"), config=config)
                    assert "__interrupt__" not in result
                    assert result.get("error") is None
                    assert result.get("approved") is True
                    assert session_is_complete(result)
                finally:
                    _close_graph(g)

    def test_resume_loads_checkpoint_from_same_db(self):
        """Simulate crash after interrupt: new graph + invoke(None) resumes."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = str(Path(tmpdir) / "cp.db")
            session_id = "v5-test-crash"
            config = {"configurable": {"thread_id": session_id}}
            g1 = g2 = None

            try:
                with patch("graph.workflow.curriculum_planner_node", _fake_planner):
                    g1 = build_graph(db_path=db_path)
                    result = g1.invoke(
                        initial_state("Learn Python", session_id),
                        config=config,
                    )
                    assert "__interrupt__" in result
                _close_graph(g1)
                g1 = None

                # New process simulation: new graph object, same DB + thread_id
                with (
                    patch("graph.workflow.curriculum_planner_node", _fake_planner),
                    patch("graph.workflow.explainer_node", _fake_explainer),
                    patch("graph.workflow.quiz_generator_node", _fake_quiz),
                    patch("graph.workflow.progress_coach_node", _fake_coach_end),
                ):
                    g2 = build_graph(db_path=db_path)
                    result = g2.invoke(None, config=config)
                    assert "__interrupt__" in result

                    result = g2.invoke(Command(resume="yes"), config=config)
                    assert result.get("approved") is True
                    assert session_is_complete(result)
            finally:
                if g1 is not None:
                    _close_graph(g1)
                if g2 is not None:
                    _close_graph(g2)

    def test_reject_replans_then_interrupt_again(self):
        calls = {"n": 0}

        def counting_planner(state: dict) -> dict:
            calls["n"] += 1
            return _fake_planner(state)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = str(Path(tmpdir) / "cp.db")
            with (
                patch("graph.workflow.curriculum_planner_node", counting_planner),
                patch("graph.workflow.explainer_node", _fake_explainer),
                patch("graph.workflow.quiz_generator_node", _fake_quiz),
                patch("graph.workflow.progress_coach_node", _fake_coach_end),
            ):
                g = build_graph(db_path=db_path)
                try:
                    session_id = "v5-test-reject"
                    config = {"configurable": {"thread_id": session_id}}

                    result = g.invoke(
                        initial_state("Learn Python", session_id),
                        config=config,
                    )
                    assert "__interrupt__" in result
                    assert calls["n"] == 1

                    result = g.invoke(Command(resume="no"), config=config)
                    assert "__interrupt__" in result
                    assert calls["n"] == 2

                    result = g.invoke(Command(resume="yes"), config=config)
                    assert "__interrupt__" not in result
                    assert calls["n"] == 2
                finally:
                    _close_graph(g)
