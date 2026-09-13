"""
Explainer agent — Version 3.

Reads study notes via MCP tools in a multi-turn tool-calling loop,
then writes a grounded explanation into state["messages"].
"""

from __future__ import annotations

import json
import os

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from graph.state import get_current_topic
from mcp_servers.filesystem_server import (
    list_study_files,
    read_study_file,
    search_notes,
)
from mcp_servers.memory_server import memory_get, memory_set

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_TOOL_ITERATIONS = 8


@tool
def tool_list_files() -> list[str]:
    """
    List all available study note files in the notes directory.
    Returns filenames like ['closures.md', 'decorators.md'].
    Call this FIRST to discover what materials exist before reading any file.
    """
    return list_study_files()


@tool
def tool_read_file(filename: str) -> str:
    """
    Read the complete content of a study note file.
    Args:
        filename: Exact filename as returned by tool_list_files().
    Returns the full file text, or an error string if not found.
    """
    return read_study_file(filename)


@tool
def tool_search_notes(query: str) -> str:
    """
    Search across all study notes for a keyword or phrase.
    Args:
        query: Search term (case-insensitive). Example: 'nonlocal', 'closure'
    Returns a JSON string with matching lines and their file locations.
    """
    results = search_notes(query)
    if not results:
        return "No matches found."
    return json.dumps(results, indent=2)


@tool
def tool_memory_get(session_id: str, key: str) -> str:
    """
    Retrieve a value from session memory.
    Args:
        session_id: The current session ID (from state).
        key: The memory key to look up.
    Returns the stored value, or 'null' if not found.
    """
    return memory_get(session_id, key)


@tool
def tool_memory_set(session_id: str, key: str, value: str) -> str:
    """
    Store a value in session memory for later agents to read.
    Args:
        session_id: The current session ID (from state).
        key: Descriptive key name.
        value: String value. Use JSON for complex data.
    """
    return memory_set(session_id, key, value)


EXPLAINER_TOOLS = [
    tool_list_files,
    tool_read_file,
    tool_search_notes,
    tool_memory_get,
    tool_memory_set,
]
TOOL_MAP = {t.name: t for t in EXPLAINER_TOOLS}

EXPLAINER_SYSTEM_PROMPT = """You are an expert tutor explaining topics to a student.

Your explanations must be grounded in the student's actual study materials.
Use the available tools to find and read relevant notes before explaining.

APPROACH (follow this sequence):
1. Call tool_list_files() to see what materials are available
2. Call tool_search_notes(topic) to find which files cover this topic
3. Call tool_read_file(filename) to read the most relevant file(s)
4. Optionally call tool_memory_get(session_id, 'explained_topics') for prior context
5. Optionally call tool_memory_set(...) to record topics covered
6. Your FINAL message (no tool calls) MUST be the full student-facing explanation

EXPLANATION FORMAT (final message only):
- Start with a real-world analogy (1-2 sentences)
- State the core concept clearly (2-3 sentences)
- Show a concrete code example (from notes when possible)
- End with one common mistake or gotcha to watch out for

CRITICAL RULES:
- Never use your final message only to say you stored something or to offer further help.
- If notes do not cover the topic well, say so briefly, then teach from the closest
  related notes and general accurate knowledge of the topic.
- tool_memory_set is for other agents — it is NOT a substitute for teaching.
"""


def execute_tool_call(tool_call: dict) -> str:
    """Execute a tool call and return the result as a string. Never raises."""
    name = tool_call["name"]
    args = tool_call["args"]
    if name not in TOOL_MAP:
        return f"Error: unknown tool '{name}'. Available: {list(TOOL_MAP.keys())}"
    try:
        result = TOOL_MAP[name].invoke(args)
        if isinstance(result, (list, dict)):
            return json.dumps(result)
        return str(result)
    except Exception as e:
        return f"Error executing {name}({args}): {type(e).__name__}: {e}"


def _looks_like_memory_ack(content: str) -> bool:
    low = (content or "").lower().strip()
    if not low:
        return True
    if len(low) < 220 and (
        "i've stored" in low
        or "i have stored" in low
        or "stored the explanation" in low
        or ("future reference" in low and "feel free" in low)
    ):
        return True
    return False


def explainer_node(state: dict) -> dict:
    """
    LangGraph node: Explainer

    Reads:  state["roadmap"], state["current_topic_index"], state["session_id"]
    Writes: state["messages"], state["last_explanation"], state["error"]
    """
    topic = get_current_topic(state)
    if topic is None:
        return {"error": "No current topic found."}

    session_id = state.get("session_id", "unknown")
    print(f"\n[Explainer] Topic: '{topic.title}'")

    llm_with_tools = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.3,
    ).bind_tools(EXPLAINER_TOOLS)
    llm_plain = ChatOpenAI(model=MODEL_NAME, temperature=0.3)

    messages = [
        SystemMessage(content=EXPLAINER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Please explain this topic to me: '{topic.title}'\n"
                f"Context: {topic.description}\n"
                f"Session ID for memory calls: {session_id}"
            )
        ),
    ]

    final_response = None

    for iteration in range(MAX_TOOL_ITERATIONS):
        print(f"[Explainer] LLM call {iteration + 1}/{MAX_TOOL_ITERATIONS}...")
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            final_response = response
            print(f"[Explainer] Complete after {iteration + 1} LLM call(s)")
            break

        print(f"[Explainer] {len(tool_calls)} tool call(s) requested:")
        for tool_call in tool_calls:
            print(f"  -> {tool_call['name']}({tool_call['args']})")
            result = execute_tool_call(tool_call)
            log_result = result[:100] + "..." if len(result) > 100 else result
            print(f"     <- {log_result}")
            messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"],
                )
            )

    if final_response is None:
        # Hit tool-call limit — force a plain explanation turn
        print("[Explainer] Max tool iterations; forcing explanation turn...")
        messages.append(
            HumanMessage(
                content=(
                    "Stop using tools. Write the full student-facing explanation now "
                    "using the format in your instructions."
                )
            )
        )
        final_response = llm_plain.invoke(messages)
        messages.append(final_response)

    content = (final_response.content or "").strip()
    if _looks_like_memory_ack(content):
        print("[Explainer] Final reply looked like a memory ack; regenerating...")
        messages.append(
            HumanMessage(
                content=(
                    "That was not an explanation. Write the full lesson now "
                    "(analogy, concept, code example, common mistake). "
                    "Do not mention storing or memory."
                )
            )
        )
        final_response = llm_plain.invoke(messages)
        messages.append(final_response)
        content = (final_response.content or "").strip()

    print(f"[Explainer] Explanation: {len(content)} characters")
    if content:
        print(f"\n{'─' * 60}")
        print(content)
        print(f"{'─' * 60}\n")

    return {
        "messages": messages,
        "last_explanation": content,
        "error": None,
    }
