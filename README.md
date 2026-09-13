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

## Current progress

| Version | Blog chapter | Status |
|---------|--------------|--------|
| 1 | When to use multiple agents + project setup | Done |
| 2 | LangGraph state, planner, graph | Done |
| 3 | MCP + Explainer | Done |
| 4 | Quiz + Progress Coach + full loop | Done |
| 5–9 | HITL deep-dive, Langfuse, eval, A2A, … | Not yet |

## Setup (Version 1)

Use **Python 3.11 or 3.12**.

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env: set OPENAI_API_KEY
```

## Run (Version 4)

```bash
python main.py "Learn Python closures and decorators from scratch"
```

Flow: Planner → approve → **Explainer → Quiz → Coach** → next topic (or END).

A full roadmap uses many OpenAI calls. Resume anytime:

```bash
python main.py --resume <session-id>
```

## Tests

```bash
pytest tests/ -v
```
