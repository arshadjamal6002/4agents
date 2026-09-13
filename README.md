# Learning Accelerator

Multi-agent study system built chapter-by-chapter from the freeCodeCamp handbook
**How to Build a Multi-Agent AI System with LangGraph, MCP, and A2A**, adapted to
use the **OpenAI API** instead of a local Ollama model.

## What this is

Four agents coordinated by LangGraph:

1. **Curriculum Planner** — goal → structured study roadmap  
2. **Explainer** — explains topics from your notes (MCP — Chapter 3)  
3. **Quiz Generator** — quizzes and grades you (Chapter 4)  
4. **Progress Coach** — adapts and routes (Chapter 4)

Plus human approval, SQLite checkpointing, MCP tools, A2A, Langfuse, and DeepEval
as later chapters land.

## Current progress

| Chapter | Status |
|---------|--------|
| 1 — When to use multiple agents + project setup | Done |
| 2 — LangGraph state, planner, graph | Done |
| 3 — MCP + Explainer | Not yet |
| 4–9 | Not yet |

## Setup (Chapter 1)

Use **Python 3.11 or 3.12** (3.14 may lack wheels for pinned deps like `pydantic-core`).

```bash
# Windows (pick 3.12 if you have multiple Pythons):
py -3.12 -m venv .venv
.venv\Scripts\activate

# macOS/Linux:
# python3.12 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env
# Edit .env: set OPENAI_API_KEY
```

## Run (Chapter 2)

```bash
python main.py "Learn Python closures and decorators from scratch"
```

You should see a generated roadmap, then an approval prompt (`yes` / `no`).
After approval, later agents are stubs until Chapters 3–4.

Resume a session:

```bash
python main.py --resume <session-id>
```

## Tests

```bash
pytest tests/ -v
```
