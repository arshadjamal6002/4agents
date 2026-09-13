"""
Quiz Generator — Version 4.

Two LLM calls with different temperatures:
  - generate questions (creative, ~0.4)
  - grade each answer (analytical, ~0.1)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from graph.state import QuizQuestion, QuizResult, get_current_topic

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GENERATION_PROMPT = """You are a quiz designer for a student learning programming.

Given a topic and explanation, generate {n} quiz questions that test
genuine understanding, not just the ability to repeat memorized phrases.

Good questions require the student to:
  - Apply a concept to a new situation
  - Explain WHY something works, not just WHAT it does
  - Identify edge cases or common mistakes
  - Compare related concepts

Return ONLY valid JSON with no prose or markdown:
{{
  "questions": [
    {{
      "question": "Clear, specific question text ending with ?",
      "expected_answer": "Model answer in 1-3 sentences",
      "difficulty": "easy|medium|hard"
    }}
  ]
}}

Rules:
  - Include at least one question about a common mistake or gotcha
  - expected_answer should be concise but complete
  - Avoid yes/no questions. Ask for explanation or demonstration
"""

GRADING_PROMPT = """You are a fair teacher grading a student's answer.

Question: {question}
Model answer: {expected_answer}
Student's answer: {student_answer}

Grade the student's answer honestly. Be generous with partial credit:
  - Fundamentally correct with minor gaps: 0.7-0.9
  - Correct concept but imprecise: 0.5-0.7
  - Partially correct: 0.3-0.5
  - Fundamentally wrong: 0.0-0.2

Return ONLY valid JSON with no prose or markdown:
{{
  "correct": true,
  "score": 0.85,
  "feedback": "One specific sentence of feedback",
  "missing_concept": "Key concept missed, or empty string if answer is correct"
}}
"""


def _json_llm(temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=temperature,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def generate_questions(topic: str, explanation: str, n: int = 3) -> list[dict]:
    """Generate n quiz questions from the Explainer's output."""
    llm = _json_llm(temperature=0.4)
    prompt = GENERATION_PROMPT.format(n=n)
    try:
        response = llm.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(
                    content=f"Topic: {topic}\n\nExplanation:\n{explanation}"
                ),
            ]
        )
        data = json.loads(response.content)
        questions = data.get("questions", [])
        if questions and isinstance(questions, list):
            return questions
    except Exception as e:
        print(f"[Quiz Generator] Question generation failed: {e}")

    return [
        {
            "question": (
                f"In your own words, explain the key concept of {topic} "
                "and why it matters."
            ),
            "expected_answer": (
                "A clear explanation demonstrating conceptual understanding."
            ),
            "difficulty": "medium",
        }
    ]


def grade_answer(question: str, expected: str, student_answer: str) -> dict:
    """Grade a student's answer using the LLM as judge."""
    llm = _json_llm(temperature=0.1)
    prompt = GRADING_PROMPT.format(
        question=question,
        expected_answer=expected,
        student_answer=student_answer,
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return json.loads(response.content)
    except Exception as e:
        print(f"[Quiz Generator] Grading failed: {e}")
        return {
            "correct": False,
            "score": 0.5,
            "feedback": "Could not grade automatically. Please review manually.",
            "missing_concept": "",
        }


def parse_grade_payload(grade: dict) -> dict:
    """Normalize grader JSON for tests and run_quiz."""
    score = float(grade.get("score", 0.0))
    score = max(0.0, min(1.0, score))
    return {
        "correct": bool(grade.get("correct", False)),
        "score": score,
        "feedback": str(grade.get("feedback", "")),
        "missing_concept": str(grade.get("missing_concept", "") or ""),
    }


def extract_explanation(messages: list) -> str:
    """Last AIMessage with content and no tool_calls (Explainer final reply)."""
    for msg in reversed(messages or []):
        if isinstance(msg, AIMessage) and msg.content:
            tool_calls = getattr(msg, "tool_calls", None) or []
            if not tool_calls:
                return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def run_quiz(topic: str, explanation: str) -> QuizResult:
    """Interactive terminal quiz: generate → answer → grade."""
    print(f"\n{'=' * 60}")
    print(f"Quiz: {topic}")
    print(f"{'=' * 60}")
    print("Answer each question in your own words. Press Enter to submit.\n")

    questions_data = generate_questions(topic, explanation, n=3)
    graded_questions: list[QuizQuestion] = []
    total_score = 0.0
    weak_areas: list[str] = []

    for i, q_data in enumerate(questions_data, 1):
        question_text = q_data.get("question", "Explain this topic.")
        expected = q_data.get("expected_answer", "")
        difficulty = q_data.get("difficulty", "medium")

        print(f"Question {i} [{difficulty}]: {question_text}")
        user_answer = input("Your answer: ").strip()
        if not user_answer:
            user_answer = "(no answer provided)"

        print("Grading...")
        grade = parse_grade_payload(grade_answer(question_text, expected, user_answer))

        score = grade["score"]
        correct = grade["correct"]
        feedback = grade["feedback"]
        missing = grade["missing_concept"]

        total_score += score
        status = "OK" if correct else "X"
        print(f"{status} Score: {score:.0%}. {feedback}\n")

        if missing:
            weak_areas.append(missing)

        graded_questions.append(
            QuizQuestion(
                question=question_text,
                expected_answer=expected,
                user_answer=user_answer,
                correct=correct,
                feedback=feedback,
                score=score,
            )
        )

    avg_score = total_score / len(questions_data) if questions_data else 0.0
    correct_count = sum(1 for q in graded_questions if q.correct)

    print(f"{'=' * 60}")
    print(
        f"Quiz complete! Score: {avg_score:.0%} "
        f"({correct_count}/{len(graded_questions)} correct)"
    )
    if weak_areas:
        print(f"Areas to review: {', '.join(set(weak_areas))}")
    print(f"{'=' * 60}\n")

    return QuizResult(
        topic=topic,
        questions=graded_questions,
        score=avg_score,
        weak_areas=list(set(weak_areas)),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def quiz_generator_node(state: dict) -> dict:
    """
    LangGraph node: Quiz Generator

    Reads:  roadmap, current_topic_index, messages
    Writes: quiz_results, weak_areas, error (+ forward keys for resume safety)
    """
    topic = get_current_topic(state)
    if topic is None:
        return {"error": "No current topic. Curriculum Planner must run first"}

    explanation = extract_explanation(state.get("messages", []))
    if not explanation:
        print(
            "[Quiz Generator] Warning: no explanation found, "
            "generating generic quiz"
        )
        explanation = f"Topic: {topic.title}. {topic.description}"

    print(f"\n[Quiz Generator] Generating quiz for: '{topic.title}'")
    quiz_result = run_quiz(topic.title, explanation)

    existing_results = list(state.get("quiz_results", []))
    all_weak_areas = list(
        set(state.get("weak_areas", []) + quiz_result.weak_areas)
    )

    return {
        "quiz_results": existing_results + [quiz_result],
        "weak_areas": all_weak_areas,
        "error": None,
        "roadmap": state.get("roadmap"),
        "current_topic_index": state.get("current_topic_index", 0),
        "session_id": state.get("session_id", ""),
    }
