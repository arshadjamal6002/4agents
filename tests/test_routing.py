"""Routing helpers — pure Python, no LLM."""

from graph.state import StudyRoadmap, Topic, initial_state
from graph.workflow import route_after_approval, route_after_coach


def test_route_after_approval_yes():
    state = initial_state("g", "s")
    state["approved"] = True
    assert route_after_approval(state) == "explainer"


def test_route_after_approval_no():
    state = initial_state("g", "s")
    state["approved"] = False
    assert route_after_approval(state) == "curriculum_planner"


def test_route_after_coach_loop_or_end():
    state = initial_state("g", "s")
    state["roadmap"] = StudyRoadmap(
        goal="g",
        total_weeks=1,
        topics=[Topic("A", "d", 30), Topic("B", "d", 30)],
    )
    state["current_topic_index"] = 0
    assert route_after_coach(state) == "explainer"
    state["current_topic_index"] = 2
    assert route_after_coach(state) == "end"
