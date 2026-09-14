"""
Streamlit UI for the Learning Accelerator.

Same LangGraph as main.py — Streamlit widgets replace terminal input().

Local:
  streamlit run streamlit_app.py

Deploy: Streamlit Community Cloud (see README). Not Vercel.
"""

from __future__ import annotations

import atexit
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

load_dotenv()

# Short demos for team lead / tracing (override in secrets or env)
os.environ.setdefault("DEMO_MAX_TOPICS", "2")

import streamlit as st

# Must be the first Streamlit command (before st.secrets / session_state).
st.set_page_config(
    page_title="Learning Accelerator",
    page_icon="📚",
    layout="wide",
)

from langgraph.types import Command

from agents.quiz_generator import (
    extract_explanation,
    generate_questions,
    grade_answer,
    parse_grade_payload,
)
from graph.checkpointing import coerce_roadmap, list_session_ids, summarize_checkpoint_values
from graph.state import QuizQuestion, QuizResult, StudyRoadmap, initial_state
from graph.workflow import build_graph
from observability.langfuse_setup import flush_langfuse, get_run_config

UI_CHECKPOINT_DB = os.getenv("CHECKPOINT_DB_UI", "data/checkpoints_ui.db")

DEMO_PRESETS = [
    "Learn Python functions and first-class objects",
    "Learn Python closures and decorators from scratch",
    "Learn Python control structures: if/else and loops",
]


def _load_streamlit_secrets_into_env() -> None:
    """Map st.secrets → os.environ for OpenAI / Langfuse on Streamlit Cloud.

    Local: only read secrets when a secrets.toml exists (avoids the
    "No secrets found" banner; `.env` via load_dotenv is enough).
    Cloud: also try when Streamlit reports a cloud runtime.
    """
    candidates = (
        Path(__file__).parent / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    )
    on_cloud = os.getenv("STREAMLIT_RUNTIME_ENVIRONMENT", "").lower() in {
        "cloud",
        "streamlit_cloud",
    }
    if not any(path.is_file() for path in candidates) and not on_cloud:
        return
    try:
        secrets = dict(st.secrets)
    except Exception:
        return
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "LANGFUSE_BASE_URL",
        "DEMO_MAX_TOPICS",
        "NOTES_PATH",
        "CHECKPOINT_DB",
    ):
        value = secrets.get(key)
        if value:
            os.environ[key] = str(value)


_load_streamlit_secrets_into_env()
atexit.register(flush_langfuse)

ui_graph = build_graph(
    db_path=UI_CHECKPOINT_DB,
    interrupt_before=["quiz_generator"],
)


def init_state() -> None:
    defaults = {
        "screen": "GOAL_INPUT",
        "session_id": None,
        "graph_config": None,
        "roadmap": None,
        "current_topic_index": 0,
        "quiz_questions": [],
        "current_question_idx": 0,
        "graded_answers": [],
        "current_quiz_missing_concepts": [],
        "quiz_results": [],
        "weak_areas": [],
        "explanation": "",
        "topic_title": "",
        "topic_description": "",
        "coaching_message": "",
        "topic_lessons": [],  # in-session lesson cache for history
        "history_snapshot": None,  # past session summary view
        "error": None,
        "goal": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def go_to(screen: str) -> None:
    st.session_state.screen = screen


def get_roadmap() -> StudyRoadmap | None:
    return coerce_roadmap(st.session_state.roadmap)


def cache_topic_lesson(
    *,
    index: int,
    title: str,
    description: str,
    explanation: str,
    quiz_result: QuizResult | None = None,
) -> None:
    """Upsert a topic lesson so users can reopen earlier topics in this session."""
    lessons = list(st.session_state.get("topic_lessons") or [])
    entry = {
        "index": index,
        "title": title,
        "description": description,
        "explanation": explanation,
        "quiz_result": quiz_result.to_dict() if quiz_result else None,
    }
    for i, existing in enumerate(lessons):
        if existing.get("index") == index:
            if quiz_result is None and existing.get("quiz_result"):
                entry["quiz_result"] = existing["quiz_result"]
            lessons[i] = entry
            st.session_state.topic_lessons = lessons
            return
    lessons.append(entry)
    st.session_state.topic_lessons = lessons


def render_topic_history() -> None:
    lessons = st.session_state.get("topic_lessons") or []
    if not lessons:
        return
    with st.expander("Earlier topics in this session", expanded=False):
        for lesson in sorted(lessons, key=lambda x: x.get("index", 0)):
            title = lesson.get("title") or f"Topic {lesson.get('index', 0) + 1}"
            qr = lesson.get("quiz_result")
            score_bit = ""
            if qr:
                score_bit = f" — quiz {float(qr.get('score', 0)):.0%}"
            st.markdown(f"**{lesson.get('index', 0) + 1}. {title}{score_bit}**")
            if lesson.get("description"):
                st.caption(lesson["description"])
            if lesson.get("explanation"):
                st.markdown(lesson["explanation"])
            else:
                st.warning("No saved explanation for this topic.")
            if qr:
                st.markdown(f"**Score:** {float(qr.get('score', 0)):.0%}")
                weak = qr.get("weak_areas") or []
                if weak:
                    st.markdown(f"**Review:** {', '.join(weak[:4])}")
            st.markdown("---")


def resolve_explanation(result: dict, messages: list) -> str:
    text = (result.get("last_explanation") or "").strip()
    if text and "i've stored" not in text.lower():
        return text
    return extract_explanation(messages)


def resolve_coaching(result: dict) -> str:
    return (result.get("last_coaching_message") or "").strip()


def get_topic_info(result: dict, idx: int) -> tuple[str, str]:
    roadmap = coerce_roadmap(result.get("roadmap") or st.session_state.roadmap)
    if roadmap and idx < len(roadmap.topics):
        topic = roadmap.topics[idx]
        return topic.title, topic.description
    return "", ""


def list_past_session_summaries() -> list[dict]:
    ids = list_session_ids(UI_CHECKPOINT_DB)
    summaries = []
    for sid in reversed(ids[-20:]):  # newest-ish last in DB order; show recent first
        try:
            snap = ui_graph.get_state({"configurable": {"thread_id": sid}})
            summary = summarize_checkpoint_values(snap.values if snap else {})
            if not summary.get("session_id"):
                summary["session_id"] = sid
            if summary.get("goal") or summary.get("quiz_results"):
                summaries.append(summary)
        except Exception:
            continue
    return summaries


def open_past_session(summary: dict) -> None:
    st.session_state.history_snapshot = summary
    go_to("SESSION_HISTORY")


def new_session() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()


def start_session(goal: str) -> None:
    session_id = str(uuid.uuid4())[:8]
    config = get_run_config(session_id)
    st.session_state.session_id = session_id
    st.session_state.graph_config = config
    st.session_state.goal = goal
    st.session_state.topic_lessons = []
    st.session_state.history_snapshot = None

    state = initial_state(goal, session_id)
    with st.spinner("Building your study roadmap..."):
        result = ui_graph.invoke(state, config=config)

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        st.session_state.roadmap = payload.get("roadmap")
        go_to("ROADMAP_APPROVAL")
    elif result.get("error"):
        st.session_state.error = result["error"]
    else:
        st.session_state.error = "Unexpected: no interrupt after planner."


def approve_roadmap(approved: bool) -> None:
    decision = "yes" if approved else "no"
    with st.spinner(
        "Starting your study session..." if approved else "Generating a new plan..."
    ):
        result = ui_graph.invoke(
            Command(resume=decision),
            config=st.session_state.graph_config,
        )

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        st.session_state.roadmap = payload.get("roadmap")
        go_to("ROADMAP_APPROVAL")
        return

    messages = result.get("messages", [])
    explanation = resolve_explanation(result, messages)
    st.session_state.explanation = explanation
    st.session_state.coaching_message = ""
    st.session_state.roadmap = result.get("roadmap") or st.session_state.roadmap
    idx = result.get("current_topic_index", 0)
    st.session_state.current_topic_index = idx
    title, desc = get_topic_info(result, idx)
    st.session_state.topic_title = title
    st.session_state.topic_description = desc
    cache_topic_lesson(
        index=idx, title=title, description=desc, explanation=explanation
    )

    with st.spinner("Generating quiz questions..."):
        st.session_state.quiz_questions = generate_questions(
            title, explanation or f"{title}. {desc}", n=3
        )
    st.session_state.current_question_idx = 0
    st.session_state.graded_answers = []
    st.session_state.current_quiz_missing_concepts = []
    go_to("EXPLAINING")


def advance_after_quiz(quiz_result: QuizResult) -> None:
    config = st.session_state.graph_config
    existing = list(st.session_state.quiz_results)
    all_weak = list(set(st.session_state.weak_areas + quiz_result.weak_areas))

    cache_topic_lesson(
        index=st.session_state.current_topic_index,
        title=st.session_state.topic_title,
        description=st.session_state.topic_description,
        explanation=st.session_state.explanation,
        quiz_result=quiz_result,
    )

    ui_graph.update_state(
        config,
        {
            "quiz_results": existing + [quiz_result],
            "weak_areas": all_weak,
            "roadmap": st.session_state.roadmap,
            "current_topic_index": st.session_state.current_topic_index,
            "session_id": st.session_state.session_id,
            "error": None,
        },
        as_node="quiz_generator",
    )

    with st.spinner("Getting coaching feedback..."):
        result = ui_graph.invoke(None, config=config)

    messages = result.get("messages", [])
    st.session_state.coaching_message = resolve_coaching(result)
    st.session_state.quiz_results = result.get(
        "quiz_results", existing + [quiz_result]
    )
    st.session_state.weak_areas = result.get("weak_areas", all_weak)
    new_idx = result.get(
        "current_topic_index", st.session_state.current_topic_index + 1
    )
    st.session_state.current_topic_index = new_idx
    st.session_state.roadmap = result.get("roadmap", st.session_state.roadmap)

    rm = get_roadmap()
    if rm is None or new_idx >= len(rm.topics):
        flush_langfuse()
        go_to("COMPLETE")
        return

    explanation = resolve_explanation(result, messages)
    st.session_state.explanation = explanation
    title, desc = get_topic_info(result, new_idx)
    st.session_state.topic_title = title
    st.session_state.topic_description = desc
    cache_topic_lesson(
        index=new_idx, title=title, description=desc, explanation=explanation
    )

    with st.spinner("Generating quiz questions..."):
        st.session_state.quiz_questions = generate_questions(
            title, explanation or f"{title}. {desc}", n=3
        )
    st.session_state.current_question_idx = 0
    st.session_state.graded_answers = []
    st.session_state.current_quiz_missing_concepts = []
    go_to("EXPLAINING")


def screen_goal_input() -> None:
    _, mid, _ = st.columns([1, 2.2, 1])
    with mid:
        st.title("Learning Accelerator")
        st.markdown(
            "Enter a learning goal. The system builds a study plan, explains "
            "topics from your notes (MCP), quizzes you, and coaches you — "
            "powered by **LangGraph + OpenAI**."
        )
        demo_n = os.getenv("DEMO_MAX_TOPICS", "2")
        st.caption(
            f"Demo mode: roadmap trimmed to {demo_n} topic(s) (`DEMO_MAX_TOPICS`). "
            "Prefer a preset that matches the sample notes."
        )

        st.markdown("**Quick start (matches sample notes)**")
        for i, preset in enumerate(DEMO_PRESETS):
            if st.button(preset, key=f"preset_{i}", use_container_width=True):
                start_session(preset)
                st.rerun()

        st.markdown("---")
        with st.form("goal_form"):
            goal = st.text_input(
                "Or type your own goal",
                placeholder="e.g. Learn Python closures and decorators from scratch",
            )
            submitted = st.form_submit_button("Build study plan", type="primary")

        if submitted:
            if not goal.strip():
                st.error("Please enter a learning goal.")
            else:
                start_session(goal.strip())
                st.rerun()

        if st.session_state.error:
            st.error(f"Error: {st.session_state.error}")
            if st.button("Try again"):
                st.session_state.error = None
                st.rerun()

        st.markdown("---")
        st.subheader("Earlier sessions")
        past = list_past_session_summaries()
        if not past:
            st.caption("No saved sessions yet. Complete a run to see history here.")
        else:
            for summary in past[:12]:
                sid = summary.get("session_id", "?")
                goal_text = summary.get("goal") or "(no goal)"
                avg = summary.get("average_score")
                avg_text = f"{avg:.0%}" if avg is not None else "—"
                status = "complete" if summary.get("complete") else "in progress"
                label = f"`{sid}` · {goal_text[:60]} · avg {avg_text} · {status}"
                if st.button(label, key=f"hist_{sid}", use_container_width=True):
                    open_past_session(summary)
                    st.rerun()


def screen_roadmap_approval() -> None:
    _, mid, _ = st.columns([1, 2.2, 1])
    with mid:
        st.title("Your study plan")
        rm = get_roadmap()
        if rm is None:
            st.error("No roadmap found.")
            if st.button("Start over"):
                new_session()
                st.rerun()
            return

        st.markdown(f"**Goal:** {rm.goal}")
        st.markdown(
            f"**Duration:** {rm.total_weeks} weeks @ {rm.weekly_hours} hrs/week"
        )
        st.markdown("---")
        for i, topic in enumerate(rm.topics, 1):
            prereq_text = (
                f" *(needs: {', '.join(topic.prerequisites)})*"
                if topic.prerequisites
                else ""
            )
            st.markdown(
                f"**{i}. {topic.title}** — {topic.estimated_minutes} min{prereq_text}"
            )
            st.markdown(f"{topic.description}")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "Yes, start studying", type="primary", use_container_width=True
            ):
                approve_roadmap(True)
                st.rerun()
        with col2:
            if st.button("No, different plan", use_container_width=True):
                approve_roadmap(False)
                st.rerun()


def render_study_reference() -> None:
    """Left-pane study notes while quizzing (read-only reference)."""
    st.markdown(f"### {st.session_state.topic_title or 'Study notes'}")
    if st.session_state.topic_description:
        st.caption(st.session_state.topic_description)
    st.markdown("---")
    if st.session_state.explanation:
        st.markdown(st.session_state.explanation)
    else:
        st.warning("No explanation available for this topic.")


def screen_explaining() -> None:
    # Keep pre-quiz reading comfortable on wide layout
    _, mid, _ = st.columns([1, 2.2, 1])
    with mid:
        rm = get_roadmap()
        total = len(rm.topics) if rm else 1
        idx = st.session_state.current_topic_index
        st.progress(min(idx / max(total, 1), 1.0), text=f"Topic {idx + 1} of {total}")
        st.title(st.session_state.topic_title)
        st.caption(st.session_state.topic_description)
        render_topic_history()
        st.markdown("---")

        if st.session_state.coaching_message:
            st.info(f"**Coach:** {st.session_state.coaching_message}")
            st.markdown("---")

        if st.session_state.explanation:
            st.markdown("### Explanation")
            st.markdown(st.session_state.explanation)
        else:
            st.warning(
                "No explanation available — notes may not cover this topic. "
                "Quiz will use the topic title/description."
            )

        st.markdown("---")
        if st.button("Start quiz", type="primary"):
            st.session_state.coaching_message = ""
            go_to("QUIZZING")
            st.rerun()


def screen_quizzing() -> None:
    """Split view: study notes left, quiz right (reference while answering)."""
    questions = st.session_state.quiz_questions
    q_idx = st.session_state.current_question_idx
    total_q = len(questions)
    rm = get_roadmap()
    total_topics = len(rm.topics) if rm else 1
    topic_idx = st.session_state.current_topic_index

    st.markdown(
        """
        <style>
        div[data-testid="stHorizontalBlock"] > div:first-child {
            border-right: 1px solid rgba(250, 250, 250, 0.12);
            padding-right: 1.25rem;
        }
        div[data-testid="stHorizontalBlock"] > div:last-child {
            padding-left: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.caption("Reference")
        render_topic_history()
        render_study_reference()

    with right:
        st.progress(
            min(topic_idx / max(total_topics, 1), 1.0),
            text=f"Topic {topic_idx + 1} of {total_topics}",
        )
        if total_q > 0:
            st.progress(
                min(q_idx / max(total_q, 1), 1.0),
                text=f"Question {min(q_idx + 1, total_q)} of {total_q}",
            )

        st.title("Quiz")
        st.markdown("---")

        for i, graded in enumerate(st.session_state.graded_answers):
            status = "OK" if graded.correct else "X"
            with st.expander(
                f"{status} Q{i + 1}: {graded.question[:80]}...", expanded=False
            ):
                st.markdown(f"**Your answer:** {graded.user_answer}")
                st.markdown(f"**Score:** {graded.score:.0%}")
                st.markdown(f"**Feedback:** {graded.feedback}")

        if q_idx < total_q:
            q = questions[q_idx]
            question_text = q.get("question", "")
            difficulty = q.get("difficulty", "medium")
            st.markdown(f"**Question {q_idx + 1} [{difficulty}]:**")
            st.markdown(question_text)

            with st.form(f"answer_form_{q_idx}"):
                answer = st.text_area(
                    "Your answer:", height=140, key=f"answer_input_{q_idx}"
                )
                submitted = st.form_submit_button("Submit answer", type="primary")

            if submitted:
                user_answer = answer.strip() or "(no answer provided)"
                expected = q.get("expected_answer", "")
                with st.spinner("Grading..."):
                    grade = parse_grade_payload(
                        grade_answer(question_text, expected, user_answer)
                    )
                graded_q = QuizQuestion(
                    question=question_text,
                    expected_answer=expected,
                    user_answer=user_answer,
                    correct=grade["correct"],
                    feedback=grade["feedback"],
                    score=grade["score"],
                )
                st.session_state.graded_answers.append(graded_q)
                missing = grade["missing_concept"].strip()
                if missing:
                    st.session_state.current_quiz_missing_concepts.append(missing)
                st.session_state.current_question_idx = q_idx + 1
                st.rerun()
        else:
            graded = st.session_state.graded_answers
            avg_score = (
                sum(q.score for q in graded) / len(graded) if graded else 0.0
            )
            weak_areas = list(
                dict.fromkeys(st.session_state.current_quiz_missing_concepts)
            )
            st.success("Quiz complete")
            st.metric("Your score", f"{avg_score:.0%}")
            quiz_result = QuizResult(
                topic=st.session_state.topic_title,
                questions=graded,
                score=avg_score,
                weak_areas=weak_areas,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            if st.button("Continue", type="primary"):
                advance_after_quiz(quiz_result)
                st.rerun()


def screen_complete() -> None:
    _, mid, _ = st.columns([1, 2.2, 1])
    with mid:
        st.title("Session complete")
        st.markdown("---")
        rm = get_roadmap()
        quiz_results = st.session_state.quiz_results
        if rm:
            st.markdown(f"**Goal:** {rm.goal}")
        if quiz_results:
            scores = []
            for r in quiz_results:
                if isinstance(r, dict):
                    scores.append(float(r.get("score", 0)))
                else:
                    scores.append(r.score)
            avg = sum(scores) / len(scores)
            st.metric("Overall average", f"{avg:.0%}")
            st.markdown("### Results by topic")
            for r in quiz_results:
                if isinstance(r, dict):
                    r = QuizResult.from_dict(r)
                status = "OK" if r.score >= 0.5 else "X"
                weak = (
                    f" — review: {', '.join(r.weak_areas[:2])}" if r.weak_areas else ""
                )
                st.markdown(f"{status} **{r.topic}**: {r.score:.0%}{weak}")

        lessons = st.session_state.get("topic_lessons") or []
        if lessons:
            st.markdown("### Review lessons")
            for lesson in sorted(lessons, key=lambda x: x.get("index", 0)):
                title = lesson.get("title") or "Topic"
                with st.expander(f"{lesson.get('index', 0) + 1}. {title}"):
                    if lesson.get("explanation"):
                        st.markdown(lesson["explanation"])
                    else:
                        st.caption("No explanation cached.")
                    qr = lesson.get("quiz_result")
                    if qr:
                        st.markdown(f"**Quiz score:** {float(qr.get('score', 0)):.0%}")

        st.markdown("---")
        st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
        if st.button("Start a new session", type="primary"):
            new_session()
            st.rerun()


def screen_session_history() -> None:
    _, mid, _ = st.columns([1, 2.2, 1])
    with mid:
        summary = st.session_state.get("history_snapshot") or {}
        st.title("Past session")
        st.markdown(f"**Session ID:** `{summary.get('session_id', '?')}`")
        st.markdown(f"**Goal:** {summary.get('goal') or '(unknown)'}")
        status = "Complete" if summary.get("complete") else "In progress"
        st.markdown(
            f"**Status:** {status} · "
            f"topic index {summary.get('current_topic_index', 0)}/"
            f"{summary.get('topic_count', 0)}"
        )
        avg = summary.get("average_score")
        if avg is not None:
            st.metric("Average score", f"{avg:.0%}")

        quiz_results = summary.get("quiz_results") or []
        if quiz_results:
            st.markdown("### Results by topic")
            for r in quiz_results:
                if isinstance(r, dict):
                    r = QuizResult.from_dict(r)
                status_mark = "OK" if r.score >= 0.5 else "X"
                weak = (
                    f" — review: {', '.join(r.weak_areas[:2])}" if r.weak_areas else ""
                )
                st.markdown(f"{status_mark} **{r.topic}**: {r.score:.0%}{weak}")
                with st.expander(f"Answers — {r.topic}"):
                    for i, q in enumerate(r.questions, 1):
                        mark = "OK" if q.correct else "X"
                        st.markdown(f"**{mark} Q{i}:** {q.question}")
                        st.markdown(f"Your answer: {q.user_answer}")
                        st.markdown(f"Feedback: {q.feedback}")
                        st.markdown("---")

        coaching = summary.get("last_coaching_message") or ""
        if coaching:
            st.info(f"**Last coach note:** {coaching}")

        explanation = summary.get("last_explanation") or ""
        if explanation:
            with st.expander("Last saved explanation from this session"):
                st.markdown(explanation)

        st.markdown("---")
        if st.button("Back to home", type="primary"):
            st.session_state.history_snapshot = None
            go_to("GOAL_INPUT")
            st.rerun()


def display_error() -> None:
    if st.session_state.error:
        st.error(f"Something went wrong: {st.session_state.error}")
        if st.button("Start over"):
            new_session()
            st.rerun()


screen = st.session_state.screen
if screen == "GOAL_INPUT":
    screen_goal_input()
elif screen == "ROADMAP_APPROVAL":
    display_error()
    screen_roadmap_approval()
elif screen == "EXPLAINING":
    display_error()
    screen_explaining()
elif screen == "QUIZZING":
    display_error()
    screen_quizzing()
elif screen == "COMPLETE":
    screen_complete()
elif screen == "SESSION_HISTORY":
    screen_session_history()
else:
    st.error(f"Unknown screen: {screen}")
    if st.button("Reset"):
        new_session()
        st.rerun()
