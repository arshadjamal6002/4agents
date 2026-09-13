"""
Shared state for the Learning Accelerator (Chapter 2).

Every agent reads from and writes partial updates to this state.
LangGraph checkpoints it to SQLite after every node.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


@dataclass
class Topic:
    """One topic in the study roadmap."""

    title: str
    description: str
    estimated_minutes: int
    prerequisites: list[str] = field(default_factory=list)
    # pending → in_progress → completed | needs_review
    status: str = "pending"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Topic:
        return cls(
            title=data["title"],
            description=data["description"],
            estimated_minutes=data["estimated_minutes"],
            prerequisites=data.get("prerequisites", []),
            status=data.get("status", "pending"),
        )


@dataclass
class StudyRoadmap:
    """Full study plan from the Curriculum Planner."""

    goal: str
    total_weeks: int
    topics: list[Topic]
    weekly_hours: int = 5

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "total_weeks": self.total_weeks,
            "weekly_hours": self.weekly_hours,
            "topics": [t.to_dict() for t in self.topics],
        }

    @classmethod
    def from_dict(cls, data: dict) -> StudyRoadmap:
        return cls(
            goal=data["goal"],
            total_weeks=data["total_weeks"],
            weekly_hours=data.get("weekly_hours", 5),
            topics=[Topic.from_dict(t) for t in data.get("topics", [])],
        )

    def completed_count(self) -> int:
        return sum(1 for t in self.topics if t.status == "completed")

    def is_complete(self) -> bool:
        return all(t.status in ("completed", "needs_review") for t in self.topics)


@dataclass
class QuizQuestion:
    """One quiz question plus grading fields filled after the user answers."""

    question: str
    expected_answer: str
    user_answer: str = ""
    correct: bool = False
    feedback: str = ""
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> QuizQuestion:
        return cls(
            question=data.get("question", ""),
            expected_answer=data.get("expected_answer", ""),
            user_answer=data.get("user_answer", ""),
            correct=bool(data.get("correct", False)),
            feedback=data.get("feedback", ""),
            score=float(data.get("score", 0.0)),
        )


@dataclass
class QuizResult:
    """Result of one quiz session on a single topic."""

    topic: str
    questions: list[QuizQuestion]
    score: float
    weak_areas: list[str]
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "score": self.score,
            "weak_areas": self.weak_areas,
            "timestamp": self.timestamp,
            "questions": [q.to_dict() for q in self.questions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> QuizResult:
        questions_raw = data.get("questions", [])
        questions = [
            QuizQuestion.from_dict(q) if isinstance(q, dict) else q
            for q in questions_raw
        ]
        return cls(
            topic=data.get("topic", ""),
            questions=questions,
            score=float(data.get("score", 0.0)),
            weak_areas=data.get("weak_areas", []),
            timestamp=data.get("timestamp", ""),
        )

    def passed(self) -> bool:
        return self.score >= 0.5

    def strong_pass(self) -> bool:
        return self.score >= 0.75


class AgentState(TypedDict):
    """
    Shared LangGraph state.

    Nodes return only the keys they changed; LangGraph merges them.
    `messages` uses add_messages so history appends instead of replacing.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    goal: str
    roadmap: StudyRoadmap | None
    approved: bool
    current_topic_index: int
    quiz_results: list[QuizResult]
    weak_areas: list[str]
    study_materials_path: str
    error: str | None


def initial_state(
    goal: str,
    session_id: str,
    study_materials_path: str = "study_materials/sample_notes",
) -> dict:
    """Factory for a new session — always use this instead of a hand-built dict."""
    return {
        "messages": [],
        "session_id": session_id,
        "goal": goal,
        "roadmap": None,
        "approved": False,
        "current_topic_index": 0,
        "quiz_results": [],
        "weak_areas": [],
        "study_materials_path": study_materials_path,
        "error": None,
    }


def get_current_topic(state: dict) -> Topic | None:
    """Current topic, or None if the session is past the last topic."""
    roadmap = state.get("roadmap")
    if roadmap is None:
        return None

    if isinstance(roadmap, dict):
        topics_raw = roadmap.get("topics", [])
    else:
        topics_raw = roadmap.topics

    idx = state.get("current_topic_index", 0)
    if idx >= len(topics_raw):
        return None

    t = topics_raw[idx]
    if isinstance(t, dict):
        return Topic.from_dict(t)
    return t


def get_latest_quiz_result(state: dict) -> QuizResult | None:
    """Most recent quiz result (handles dicts after checkpoint resume)."""
    results = state.get("quiz_results", [])
    if not results:
        return None
    latest = results[-1]
    if isinstance(latest, dict):
        return QuizResult.from_dict(latest)
    return latest


def session_is_complete(state: dict) -> bool:
    """True when current_topic_index is past the last topic."""
    roadmap = state.get("roadmap")
    if roadmap is None:
        return True
    topics = roadmap.get("topics", []) if isinstance(roadmap, dict) else roadmap.topics
    idx = state.get("current_topic_index", 0)
    return idx >= len(topics)
