# Learning Accelerator

Multi-agent study system built **version by version** from the freeCodeCamp handbook
**How to Build a Multi-Agent AI System with LangGraph, MCP, and A2A**, adapted to
use the **OpenAI API** instead of a local Ollama model.

Blog chapter N = **Version N** in this repo.

## What this is

Four agents coordinated by LangGraph:

1. **Curriculum Planner** — goal → structured study roadmap  
2. **Explainer** — explains topics from your notes via MCP  
3. **Quiz Generator** — quizzes you and grades answers (LLM-as-judge)  
4. **Progress Coach** — feedback, memory, advance / end routing  

Plus **human approval**, **SQLite checkpoints**, and **session resume**.

## Current progress

| Version | Blog chapter | Status |
|---------|--------------|--------|
| 1 | When to use multiple agents + project setup | Done |
| 2 | LangGraph state, planner, graph | Done |
| 3 | MCP + Explainer | Done |
| 4 | Quiz + Progress Coach + full loop | Done |
| 5 | Checkpointing + HITL interrupt/resume | Done |
| 6–9 | Langfuse, DeepEval, A2A, polish | Not yet |

## Setup

Use **Python 3.11 or 3.12**.

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env: set OPENAI_API_KEY
```

## Run

```bash
python main.py "Learn Python closures and decorators from scratch"
```

Flow: Planner → approve (`yes`/`no`) → Explainer → Quiz → Coach → next topic…

Resume after a stop or crash (same Session ID printed at start):

```bash
python main.py --list-sessions
python main.py --resume <session-id>
```

## Tests

```bash
pytest tests/ -v
```
