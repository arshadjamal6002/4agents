"""
Progress Coach — Version 4.

Synthesizes quiz results, updates roadmap topic status, persists progress
to MCP memory, advances current_topic_index, and lets graph routing decide
whether to loop to Explainer or END.

Study Buddy via A2A arrives in Version 8 (low-score hook is a no-op for now).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from graph.state import StudyRoadmap, get_latest_quiz_result
from mcp_servers.memory_server import memory_set

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PASS_THRESHOLD = 0.5

COACHING_PROMPT = """You are an encouraging learning coach reviewing a student's quiz results.

Provide a brief, warm coaching message (2-3 sentences max) based on:
  - The topic studied
  - Their score (0.0 = 0%, 1.0 = 100%)
  - Any weak areas identified

Return ONLY valid JSON:
{{
  "summary": "2-3 sentence encouraging summary",
  "encouragement": "One short motivational sentence for next steps"
}}

Be specific. Reference the topic and any weak areas by name.
Never be discouraging. A low score means "more practice needed", not "you failed."
"""


def get_coaching_message(topic: str, score: float, weak_areas: list[str]) -> dict:
    """Ask the LLM for a personalised coaching message."""
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.4,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    context = {
        "topic": topic,
        "score_percent": f"{score:.0%}",
        "weak_areas": weak_areas if weak_areas else ["none identified"],
    }
    try:
        response = llm.invoke(
            [
                SystemMessage(content=COACHING_PROMPT),
                HumanMessage(content=json.dumps(context)),
            ]
        )
        return json.loads(response.content)
    except Exception as e:
        print(f"[Progress Coach] LLM call failed: {e}")
        return {
            "summary": f"You scored {score:.0%} on {topic}. Keep going!",
            "encouragement": "Every topic builds on the last.",
        }


def _as_roadmap(roadmap) -> StudyRoadmap | None:
    if roadmap is None:
        return None
    if isinstance(roadmap, dict):
        return StudyRoadmap.from_dict(roadmap)
    return roadmap


def _maybe_call_study_buddy(score: float, topic: str, weak_areas: list[str]) -> None:
    """Placeholder for Version 8 A2A Study Buddy on low scores."""
    if score >= PASS_THRESHOLD:
        return
    if os.getenv("USE_STUDY_BUDDY", "false").lower() in ("1", "true", "yes"):
        print(
            f"[Progress Coach] Study Buddy requested for '{topic}' "
            f"(score {score:.0%}) — A2A arrives in Version 8."
        )


def progress_coach_node(state: dict) -> dict:
    """
    LangGraph node: Progress Coach

    Reads:  quiz_results, roadmap, current_topic_index, session_id
    Writes: roadmap, current_topic_index, messages, error
    """
    latest = get_latest_quiz_result(state)
    if latest is None:
        return {"error": "No quiz results. Quiz Generator must run first"}

    roadmap = _as_roadmap(state.get("roadmap"))
    if roadmap is None:
        return {"error": "No roadmap found"}

    idx = state.get("current_topic_index", 0)
    session_id = state.get("session_id", "unknown")
    score = latest.score

    print(f"\n[Progress Coach] Topic: '{latest.topic}'")
    print(f"[Progress Coach] Score: {score:.0%}")
    if latest.weak_areas:
        print(f"[Progress Coach] Weak areas: {', '.join(latest.weak_areas)}")

    coaching = get_coaching_message(latest.topic, score, latest.weak_areas)

    if idx < len(roadmap.topics):
        new_status = "completed" if score >= PASS_THRESHOLD else "needs_review"
        roadmap.topics[idx].status = new_status

    next_idx = idx + 1
    all_done = next_idx >= len(roadmap.topics)

    memory_set(
        session_id,
        f"progress_topic_{idx}",
        json.dumps(
            {
                "topic": latest.topic,
                "score": score,
                "weak_areas": latest.weak_areas,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ),
    )

    _maybe_call_study_buddy(score, latest.topic, latest.weak_areas)

    print(f"\n{'-' * 60}")
    print(f"Coach: {coaching.get('summary', '')}")
    print(coaching.get("encouragement", ""))

    if all_done:
        results = state.get("quiz_results", [])
        scores = []
        for r in results:
            if isinstance(r, dict):
                scores.append(float(r.get("score", 0.0)))
            else:
                scores.append(r.score)
        avg = sum(scores) / max(len(scores), 1)
        print(f"\nSession complete! Average: {avg:.0%}")
    else:
        next_topic = roadmap.topics[next_idx]
        print(f"\nNext topic: '{next_topic.title}'")
    print(f"{'-' * 60}\n")

    return {
        "roadmap": roadmap,
        "current_topic_index": next_idx,
        "messages": [AIMessage(content=coaching.get("summary", ""))],
        "error": None,
    }
