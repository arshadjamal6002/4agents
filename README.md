# Learning Accelerator

Multi-agent study system built **version by version** from the freeCodeCamp handbook
[How to Build a Multi-Agent AI System with LangGraph, MCP, and A2A](https://www.freecodecamp.org/news/how-to-build-a-multi-agent-ai-system-with-langgraph-mcp-and-a2a-full-book/),
adapted to use the **OpenAI API** instead of a local Ollama model.

**Convention:** blog chapter N = **Version N** in this repo.

---

## What this is

A **Learning Accelerator**: you give a learning goal; the system plans a curriculum,
explains topics from *your* markdown notes, quizzes you, and adapts as you go.

| Agent | Role |
|-------|------|
| **Curriculum Planner** | Goal → structured JSON study roadmap (no tools) |
| **Human Approval** | Pause for `yes` / `no` via LangGraph `interrupt()` |
| **Explainer** | MCP tool loop: list / search / read notes → grounded explanation |
| **Quiz Generator** | Generate questions + LLM-as-judge grading (interactive) |
| **Progress Coach** | Coaching message, topic status, MCP memory, advance or end |

**Stack (Versions 1–6):** LangGraph · MCP · OpenAI · SQLite checkpoints · Langfuse

---

## Progress

| Version | Focus | Status |
|---------|--------|--------|
| **1** | When multi-agent is worth it + project setup | Done |
| **2** | Shared `AgentState`, Curriculum Planner, graph wiring | Done |
| **3** | Filesystem + memory MCP servers, Explainer tool loop | Done |
| **4** | Quiz Generator + Progress Coach, full topic loop | Done |
| **5** | Checkpoint helpers, HITL harden, resume / list sessions | Done |
| **6** | Langfuse observability (cloud or self-hosted) | Done |
| 7 | DeepEval quality evaluation | Not yet |
| 8 | A2A + CrewAI Study Buddy | Not yet |
| 9 | Complete system / polish | Not yet |

---

## Architecture (current)

```
START
  → Curriculum Planner
  → Human Approval          ← interrupt(); yes → continue, no → replan
  → Explainer               ← MCP filesystem + memory tools
  → Quiz Generator          ← interactive Q&A + grading
  → Progress Coach          ← status + memory + index++
  → more topics? → Explainer : END

State after every node → SQLite (thread_id = session_id)
LLM / agent / tool calls → Langfuse (if keys set in .env)
```

OpenAI replaces the book’s Ollama for all LLM calls.

---

## Project layout

```
4agents/
├── main.py                      # CLI entry: run / resume / list-sessions
├── streamlit_app.py             # Web UI (Streamlit Cloud)
├── requirements.txt
├── pyproject.toml               # pytest pythonpath = src
├── .env.example
├── .streamlit/                  # Streamlit config (+ secrets example)
├── langchain/                   # Tiny stub for langfuse import (not full langchain)
├── src/
│   ├── agents/
│   │   ├── curriculum_planner.py
│   │   ├── human_approval.py
│   │   ├── explainer.py
│   │   ├── quiz_generator.py
│   │   └── progress_coach.py
│   ├── graph/
│   │   ├── state.py             # AgentState + dataclasses
│   │   ├── workflow.py          # StateGraph + SqliteSaver
│   │   └── checkpointing.py     # session list / coerce roadmap (V5)
│   ├── mcp_servers/
│   │   ├── filesystem_server.py # notes tools + notes://index
│   │   └── memory_server.py     # session key-value memory
│   └── observability/
│       └── langfuse_setup.py    # Langfuse CallbackHandler + flush (V6)
├── study_materials/sample_notes/
│   ├── python_basics.md
│   ├── closures.md
│   └── decorators.md
├── tests/
│   ├── test_state.py
│   ├── test_curriculum_planner.py
│   ├── test_routing.py
│   ├── test_mcp_servers.py
│   ├── test_quiz_and_coach.py
│   ├── test_checkpointing.py
│   └── test_observability.py
└── data/                        # checkpoints.db created at runtime
```

---

## Requirements

- **Python 3.11 or 3.12** (3.14 often lacks wheels for pinned deps like `pydantic-core`)
- OpenAI API key
- Windows / macOS / Linux

---

## Setup

```bash
# Windows
py -3.12 -m venv .venv
.venv\Scripts\activate

# macOS / Linux
# python3.12 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux
```

Edit `.env`:

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Required |
| `OPENAI_MODEL` | Default `gpt-4o-mini` |
| `CHECKPOINT_DB` | Default `data/checkpoints.db` |
| `NOTES_PATH` | Default `study_materials/sample_notes` |
| `DEMO_MAX_TOPICS` | Optional; trim roadmap (default `2` in Streamlit) |
| `LANGFUSE_PUBLIC_KEY` / `SECRET_KEY` | Optional — leave empty to skip tracing |
| `LANGFUSE_BASE_URL` or `LANGFUSE_HOST` | Cloud: `https://cloud.langfuse.com` |
| `USE_A2A_*` / service URLs | Unused until Version 8 |

---

## Run (CLI)

```bash
python main.py "Learn Python closures and decorators from scratch"
```

**Session flow**

1. Planner builds a roadmap (printed with Session ID)
2. Type `yes` to approve or `no` to regenerate
3. Explainer reads your notes via MCP and prints an explanation
4. Quiz asks questions; you answer in the terminal; answers are graded
5. Coach gives feedback and moves to the next topic (or ends)

Set `DEMO_MAX_TOPICS=2` in `.env` for a short demo / faster Langfuse test.
`main.py` always flushes Langfuse traces in a `finally` block (including Ctrl+C).

### Web UI (Streamlit) — share a URL with your lead

**Not Vercel** — Streamlit needs a long-running Python host.

#### Local

```bash
streamlit run streamlit_app.py
```

Opens `http://localhost:8501` — approve, explain, quiz in the browser.

#### Deploy on Streamlit Community Cloud (recommended)

1. Push this repo to GitHub (public or private).
2. Go to [https://share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select the repo, branch, and main file: `streamlit_app.py`.
   (`runtime.txt` pins Python 3.12 for Cloud.)
4. Under **Advanced settings → Secrets**, paste TOML (see `.streamlit/secrets.toml.example`):

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o-mini"
LANGFUSE_PUBLIC_KEY = "pk-lf-..."
LANGFUSE_SECRET_KEY = "sk-lf-..."
LANGFUSE_BASE_URL = "https://cloud.langfuse.com"
DEMO_MAX_TOPICS = "2"
```

5. Deploy → you get a URL like `https://your-app.streamlit.app` to send your team lead.

Never commit real secrets; `.streamlit/secrets.toml` is gitignored.

### Resume (Version 5)

Every run prints a **Session ID**. After Ctrl+C or a crash:

```bash
python main.py --list-sessions
python main.py --resume <session-id>
```

Resume loads the latest SQLite checkpoint for that `thread_id` and continues
from the interrupted step (e.g. still waiting for approval, or mid-loop).

### Your own notes

Add or replace `.md` files under `study_materials/sample_notes/` (or change
`NOTES_PATH`). The Explainer discovers them through the filesystem MCP server.

### Observability — Langfuse Cloud (Version 6)

No Docker required. Create a project at [cloud.langfuse.com](https://cloud.langfuse.com),
put the keys in `.env`, then run the app. You should see:

```text
[Observability] Tracing session <id> → https://cloud.langfuse.com
```

After the first LLM call (Curriculum Planner), refresh the Langfuse UI — traces
appear within seconds. Leave keys empty to run without tracing.

---

## Tests

Unit tests do **not** need an OpenAI key (LLM calls are mocked where needed):

```bash
pytest tests/ -v
```

Coverage includes state helpers, roadmap parsing, MCP tools (path traversal,
memory), quiz/coach state updates, and interrupt → resume / reject → replan
checkpoint flows.

---

## Version changelog (implemented)

### Version 1 — Setup
Project layout, deps, `.env`, OpenAI instead of Ollama, rationale for four agents.

### Version 2 — LangGraph foundation
`AgentState`, Curriculum Planner (JSON mode), graph edges, human approval node,
SQLite checkpointer wiring.

### Version 3 — MCP + Explainer
Filesystem and memory MCP servers; Explainer multi-turn tool-calling loop
grounded in study notes.

### Version 4 — Quiz + Coach
Interactive quiz generation/grading; Progress Coach updates topic status,
writes memory, advances `current_topic_index` so the graph loops until done.

### Version 5 — Persistence & HITL
Checkpoint helpers (`list_session_ids`, `session_exists`, `coerce_roadmap`);
interrupt payload uses serializable roadmap dicts; full state re-emitted after
resume; CLI `--list-sessions` and safer `--resume`; checkpoint integration tests.

### Version 6 — Langfuse observability
LangChain `CallbackHandler` on `graph.invoke`; supports Langfuse Cloud via
`LANGFUSE_BASE_URL` (no Docker); graceful degrade when keys are missing;
`flush_langfuse()` on exit.

---

## What’s next

- **Version 7** — DeepEval LLM-as-judge quality metrics  
- **Version 8** — A2A services + CrewAI Study Buddy on low quiz scores  
- **Version 9** — End-to-end polish  

Companion reference (book’s full code, Ollama-based):
[sandeepmb/freecodecamp-multi-agent-ai-system](https://github.com/sandeepmb/freecodecamp-multi-agent-ai-system)
