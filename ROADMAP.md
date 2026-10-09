# LangGraph Learning Roadmap

### Goal: Build stateful, controllable, production-grade LLM systems

LangGraph is the layer beneath modern agents and memory. Chains (LCEL) go in a
straight line; LangGraph lets you **loop, branch, persist, and pause for humans**.
Work through these in order. Check off each as you complete it.

> Prereq: finish the `langChain/` project first (LCEL, chains, structured output,
> the basics of agents & memory). This project goes deeper into the graph engine.

---

## Stage 1 — Fundamentals

> The graph mental model: state, nodes, edges, branching, loops.

- 1. **Why LangGraph** — chains vs graphs; your first `StateGraph` (nodes/edges/state)
- 1. **State & Reducers** — how node updates MERGE; `add_messages`; per-field reducers
- 1. **Conditional Edges** — runtime branching/routing (the thing chains can't do)
- 1. **Cycles & Loops** — self-correcting loops; the `recursion_limit` backstop

## Stage 2 — Agents & State

> Build the agent loop yourself, then make state durable and human-supervised.

- 1. **Tool Agent From Scratch** — the ReAct loop with `ToolNode` + `tools_condition`
- 1. **Persistence & Checkpointing** — memory, threads, `get_state`, time travel
- 1. **Human-in-the-Loop** — pause for approval, edit state, then resume

## Stage 3 — Advanced & Production

> Streaming, multi-agent systems, and hardening for production.

- 1. **Streaming** — `stream_mode` updates / values / messages (tokens + progress)
- 1. **Multi-Agent & Subgraphs** — supervisor routing; compose graphs as nodes
- 1. **Production Patterns** — retries, error routing, debugging, when-to-use guide

---

## Milestone Projects

> Build these to cement the concepts.


| #   | Project                                                   | Covers     |
| --- | --------------------------------------------------------- | ---------- |
| A   | Self-correcting writer (generate → critique → retry loop) | Stages 1   |
| B   | Tool-using research agent with memory                     | Stages 1–2 |
| C   | Approval-gated action bot (human-in-the-loop)             | Stage 2    |
| D   | Supervisor multi-agent system, streamed + hardened        | Stages 1–3 |


---

## How to run

```bash
./run.sh                                              # once: build image + start container
./start.sh stage1_fundamentals/01_why_langgraph.py    # run any lesson
```

Each file has demos commented out in `__main__` — uncomment ONE at a time.

## Current Topic

> Update this line as you progress. Start with **Topic 1 — Why LangGraph**.

