# LangGraph Study Guide

### Your step-by-step execution plan

---

## Folder Structure

```
langgraph_deep/
├── ROADMAP.md              ← topic checklist
├── GUIDE.md                ← this file
├── README.md              ← overview
├── DOCKER.md              ← container instructions
├── .env                    ← API key (never commit)
├── .env.example
├── Dockerfile
├── requirements.txt
├── run.sh                  ← build image + start container (once)
├── start.sh                ← run any lesson file
│
├── stage1_fundamentals/
│   ├── 01_why_langgraph.py
│   ├── 02_state_and_reducers.py
│   ├── 03_conditional_edges.py
│   └── 04_cycles_and_loops.py
│
├── stage2_agents_and_state/
│   ├── 05_tool_agent_from_scratch.py
│   ├── 06_persistence_checkpointing.py
│   └── 07_human_in_the_loop.py
│
└── stage3_advanced/
    ├── 08_streaming.py
    ├── 09_multi_agent_subgraphs.py
    └── 10_production_patterns.py
```

Every file follows the same shape: a teaching docstring (concept → why → failure
modes → when to use), small `demo_*()` functions, and a `__main__` block with the
calls **commented out** so you run ONE at a time.

---

## Stage 1 — Fundamentals

### Topic 1: Why LangGraph

**Learn:** When a chain isn't enough; the node / edge / state model.
**Code:** A one-node `StateGraph`, then a two-node graph where state accumulates.
**File:** `stage1_fundamentals/01_why_langgraph.py`

### Topic 2: State & Reducers

**Learn:** How a node's partial update MERGES into state — overwrite vs combine.
**Code:** Default overwrite, an append reducer, and `add_messages` (chat state).
**File:** `stage1_fundamentals/02_state_and_reducers.py`

### Topic 3: Conditional Edges

**Learn:** Branching — let the graph pick the next node at runtime.
**Code:** A router node, an LLM intent-router, and a guardrail branch.
**File:** `stage1_fundamentals/03_conditional_edges.py`

### Topic 4: Cycles & Loops

**Learn:** Looping back — generate → evaluate → retry; bounding the loop.
**Code:** A self-correcting tagline loop + a `recursion_limit` safety demo.
**File:** `stage1_fundamentals/04_cycles_and_loops.py`

---

## Stage 2 — Agents & State

### Topic 5: Tool Agent From Scratch

**Learn:** What `create_agent` actually does — the ReAct loop as a graph.
**Code:** `agent` node + `ToolNode` + `tools_condition` wired into a cycle.
**File:** `stage2_agents_and_state/05_tool_agent_from_scratch.py`

### Topic 6: Persistence & Checkpointing

**Learn:** Durable state — memory, threads, inspection, time travel.
**Code:** `MemorySaver`, `get_state`, thread isolation, `get_state_history`.
**File:** `stage2_agents_and_state/06_persistence_checkpointing.py`

### Topic 7: Human-in-the-Loop

**Learn:** Pause before risky actions; approve / edit / resume.
**Code:** `interrupt_before`, `get_state`, `update_state`, resume with `None`.
**File:** `stage2_agents_and_state/07_human_in_the_loop.py`

---

## Stage 3 — Advanced & Production

### Topic 8: Streaming

**Learn:** Show progress and tokens from a running graph.
**Code:** `stream_mode` = updates / values / messages, plus `astream`.
**File:** `stage3_advanced/08_streaming.py`

### Topic 9: Multi-Agent & Subgraphs

**Learn:** Specialise and coordinate — supervisor routing; graphs as nodes.
**Code:** A supervisor + workers, and a compiled subgraph used as a node.
**File:** `stage3_advanced/09_multi_agent_subgraphs.py`

### Topic 10: Production Patterns

**Learn:** Hardening — retries, error routing, debugging, tool selection.
**Code:** `RetryPolicy`, error-as-state routing, `get_state`, decision guide.
**File:** `stage3_advanced/10_production_patterns.py`

---

## Rules While Studying

1. **Code every topic** — type it, don't just read.
2. **Run one demo at a time** — uncomment a single call in `__main__`.
3. **Break things on purpose** — remove `recursion_limit`, change a router, see it fail.
4. **Always bound loops** — set `recursion_limit` on anything with a cycle.
5. **Check off ROADMAP.md** after each topic.

