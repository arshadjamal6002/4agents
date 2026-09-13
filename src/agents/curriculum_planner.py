"""Curriculum Planner node — Chapter 2 (OpenAI instead of Ollama)."""

from __future__ import annotations

import json
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from graph.state import StudyRoadmap, Topic

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

PLANNER_SYSTEM_PROMPT = """You are an expert curriculum designer. Your job is to
create a structured study roadmap when given a learning goal.

Return ONLY valid JSON with no prose, no markdown code fences, no explanation.
The JSON must match this exact schema:

{
  "goal": "the original learning goal exactly as given",
  "total_weeks": <integer between 1 and 12>,
  "weekly_hours": <integer between 3 and 10>,
  "topics": [
    {
      "title": "Short topic name (3-6 words)",
      "description": "One clear sentence explaining what this topic covers",
      "estimated_minutes": <integer between 30 and 120>,
      "prerequisites": ["title of earlier topic if required, else empty list"],
      "status": "pending"
    }
  ]
}

Rules:
- Order topics from foundational to advanced
- prerequisites must reference earlier topic titles exactly as written
- Aim for 4 to 6 topics
- status must always be "pending"
"""


def build_planner_llm() -> ChatOpenAI:
    """Low temperature + JSON mode for reliable structured roadmaps."""
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.1,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def parse_roadmap_json(json_string: str) -> StudyRoadmap:
    """Parse LLM JSON into StudyRoadmap. Separated for unit tests without API calls."""
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM returned invalid JSON.\n"
            f"Error: {e}\n"
            f"Raw output (first 300 chars): {json_string[:300]}"
        ) from e

    required = ["goal", "total_weeks", "topics"]
    for field_name in required:
        if field_name not in data:
            raise ValueError(f"LLM JSON missing required field: '{field_name}'")

    if not isinstance(data["topics"], list) or len(data["topics"]) == 0:
        raise ValueError("LLM JSON 'topics' must be a non-empty list")

    topics = []
    for i, t in enumerate(data["topics"]):
        for field_name in ["title", "description", "estimated_minutes"]:
            if field_name not in t:
                raise ValueError(f"Topic {i} missing required field: '{field_name}'")
        topics.append(
            Topic(
                title=t["title"],
                description=t["description"],
                estimated_minutes=int(t["estimated_minutes"]),
                prerequisites=t.get("prerequisites", []),
                status=t.get("status", "pending"),
            )
        )

    return StudyRoadmap(
        goal=data["goal"],
        total_weeks=int(data["total_weeks"]),
        weekly_hours=int(data.get("weekly_hours", 5)),
        topics=topics,
    )


def curriculum_planner_node(state: dict) -> dict:
    """
    LangGraph node: Curriculum Planner

    Reads:  state["goal"]
    Writes: state["roadmap"], state["messages"], state["error"]
    """
    goal = state.get("goal", "").strip()
    if not goal:
        return {"error": "No learning goal provided."}

    print(f"\n[Curriculum Planner] Building roadmap for: '{goal}'")

    llm = build_planner_llm()
    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"Create a study roadmap for: {goal}"),
    ]

    print(f"[Curriculum Planner] Calling {MODEL_NAME}...")
    response = llm.invoke(messages)

    try:
        roadmap = parse_roadmap_json(response.content)
    except ValueError as e:
        print(f"[Curriculum Planner] Parse error: {e}")
        return {
            "error": str(e),
            "messages": messages + [response],
        }

    # Optional short demos: DEMO_MAX_TOPICS=2 (or MAX_TOPICS)
    max_raw = os.getenv("DEMO_MAX_TOPICS") or os.getenv("MAX_TOPICS")
    if max_raw:
        try:
            n = max(1, int(max_raw))
            if len(roadmap.topics) > n:
                roadmap.topics = roadmap.topics[:n]
                print(f"[Curriculum Planner] Trimmed to {n} topic(s) (DEMO_MAX_TOPICS)")
        except ValueError:
            pass

    print(f"[Curriculum Planner] Created {len(roadmap.topics)} topics")
    return {
        "roadmap": roadmap,
        "messages": messages + [response],
        "error": None,
    }
