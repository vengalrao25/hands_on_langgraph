# LangGraph Deep 🕸️

A hands-on, production-first course in **LangGraph** — the state-machine engine
behind modern agents and durable memory. Sibling project to `langChain/`.

> **LCEL chains** flow in a straight line (a DAG).
> **LangGraph** lets you **loop, branch, persist state, and pause for humans** —
> everything a real agentic system needs.

## Why this project exists
`langChain/` teaches you to *use* prebuilt pieces (`create_agent`, memory).
This project teaches you to *build* that machinery yourself, so nothing is magic:
the agent loop, the memory, the human approval gate — you'll construct each one.

## Quick start
```bash
cp .env.example .env          # add your OPENAI_API_KEY
./run.sh                      # build image + start container (once)
./start.sh stage1_fundamentals/01_why_langgraph.py
```
See [DOCKER.md](DOCKER.md) for full container instructions.

## What you'll learn
| Stage | Topics |
|-------|--------|
| **1 — Fundamentals** | graphs vs chains, state & reducers, conditional edges, cycles/loops |
| **2 — Agents & State** | build an agent from scratch, persistence/checkpointing, human-in-the-loop |
| **3 — Advanced** | streaming, multi-agent & subgraphs, production patterns |

Full plan in [ROADMAP.md](ROADMAP.md) · step-by-step in [GUIDE.md](GUIDE.md).

## How each lesson works
A teaching docstring (concept → why → failure modes → when to use), small
`demo_*()` functions, and a `__main__` block with calls **commented out** — run
ONE at a time so you absorb each idea.

## Stack
LangChain 1.x · LangGraph 1.x · OpenAI · Docker. Deps in [requirements.txt](requirements.txt).
