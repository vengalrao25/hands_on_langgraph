"""
LangGraph 10: Production Patterns (make a graph shippable)
----------------------------------------------------------
You can build graphs. This file is the production hardening: surviving failures,
preventing runaway cost, debugging, and knowing when LangGraph is even the right
tool.

WHAT THIS FILE COVERS:
  1. RetryPolicy — auto-retry a flaky node
  2. Error handling as STATE — catch, record, and route around failures
  3. recursion_limit — the universal loop backstop (recap, because it matters)
  4. Debugging with get_state
  5. When to use LangGraph vs create_agent vs a plain LCEL chain
"""

import random
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------------------------------------------------------
# 1. RetryPolicy — retry a node that fails transiently
# ---------------------------------------------------------------------------
def demo_retry_policy():
    """
    Nodes hit flaky APIs. Attach a RetryPolicy to a node and LangGraph re-runs it
    on failure (with backoff) before giving up — node-level resilience, separate
    from the model's own retries. Here a node fails ~half the time; the policy
    retries until it succeeds.
    """
    from langgraph.types import RetryPolicy

    class State(TypedDict):
        attempts: int
        result: str

    def flaky_node(s: State) -> dict:
        if random.random() < 0.6:
            raise ConnectionError("transient upstream failure")
        return {"result": "success"}

    b = StateGraph(State)
    b.add_node("flaky", flaky_node, retry_policy=RetryPolicy(max_attempts=5))
    b.add_edge(START, "flaky")
    b.add_edge("flaky", END)
    app = b.compile()

    print("=== RetryPolicy ===")
    try:
        out = app.invoke({"attempts": 0, "result": ""})
        print("Node eventually:", out["result"], "(retried through transient failures)")
    except Exception as e:
        print(f"Still failed after retries: {type(e).__name__}")
    print()


# ---------------------------------------------------------------------------
# 2. ERROR HANDLING AS STATE — catch, record, route around
# ---------------------------------------------------------------------------
def demo_error_as_state():
    """
    For errors you can't just retry away, catch them INSIDE the node, write an
    'error' into state, and use a conditional edge to route to a fallback path
    instead of crashing the whole graph. The graph stays alive and degrades
    gracefully — the same mindset as Stage 6 error handling, expressed as routing.
    """
    class State(TypedDict):
        query: str
        data: str
        error: str
        answer: str

    def fetch(s: State) -> dict:
        try:
            raise TimeoutError("data source unavailable")   # simulate failure
        except Exception as e:
            return {"error": str(e)}

    def route(s: State) -> str:
        return "fallback" if s.get("error") else "answer"

    def answer(s: State) -> dict:
        return {"answer": f"Here is the data: {s['data']}"}

    def fallback(s: State) -> dict:
        return {"answer": "Our data source is temporarily down — please try again later."}

    b = StateGraph(State)
    b.add_node("fetch", fetch); b.add_node("answer", answer); b.add_node("fallback", fallback)
    b.add_edge(START, "fetch")
    b.add_conditional_edges("fetch", route, {"answer": "answer", "fallback": "fallback"})
    b.add_edge("answer", END); b.add_edge("fallback", END)
    app = b.compile()

    print("=== error handling as state ===")
    out = app.invoke({"query": "sales numbers", "data": "", "error": "", "answer": ""})
    print("error :", out["error"])
    print("answer:", out["answer"], "(routed to fallback, no crash)")
    print()


# ---------------------------------------------------------------------------
# 3. recursion_limit — the universal backstop
# ---------------------------------------------------------------------------
def demo_recursion_limit():
    """
    Said before, repeated because it's the #1 production safety setting: any graph
    that can loop MUST run with a recursion_limit so a bug can't spin forever and
    drain your budget. Set it to the most steps a legitimate run could need.
    """
    print("=== recursion_limit (always set it) ===")
    print("app.invoke(input, config={'recursion_limit': 25})")
    print("-> raises GraphRecursionError instead of looping forever.")
    print()


# ---------------------------------------------------------------------------
# 4. DEBUGGING WITH get_state
# ---------------------------------------------------------------------------
def demo_debug_state():
    """
    With a checkpointer, get_state(config) shows the exact state and the next node
    — your window into 'why did it do that?'. Combined with get_state_history
    (file 06) you can replay any past step. Pair with LangSmith tracing for the
    full picture in production.
    """
    from langgraph.checkpoint.memory import MemorySaver

    class State(TypedDict):
        n: int

    def step(s: State) -> dict:
        return {"n": s["n"] + 1}

    b = StateGraph(State)
    b.add_node("step", step)
    b.add_edge(START, "step"); b.add_edge("step", END)
    app = b.compile(checkpointer=MemorySaver())

    cfg = {"configurable": {"thread_id": "dbg"}}
    app.invoke({"n": 0}, cfg)
    snap = app.get_state(cfg)
    print("=== debugging with get_state ===")
    print("values:", snap.values, "| next:", snap.next or "(finished)")
    print()


# ---------------------------------------------------------------------------
# 5. WHEN TO USE WHAT (the decision you'll make constantly)
# ---------------------------------------------------------------------------
def demo_decision_guide():
    """
    LCEL CHAIN (prompt | llm | parser):
      - fixed, one-directional steps. No loops, no branching, no shared state.
      - simplest, cheapest, most debuggable. DEFAULT to this.

    create_agent (Stage 5):
      - you need tool-calling with the standard ReAct loop, and the prebuilt
        behaviour fits. Fast to build; less control over the internals.

    RAW LANGGRAPH (this folder):
      - you need custom control flow: loops, conditional branching, multiple
        agents, human-in-the-loop, durable state, time travel, or a bespoke agent
        loop. Maximum power and control; more to build and maintain.

    PROGRESSION: start with a chain. If you need tools, try create_agent. If you
    hit its limits (custom loops/branching/HITL/multi-agent), drop to LangGraph.
    Don't reach for the most powerful tool first — reach for the simplest one that
    actually solves the problem.
    """
    print(demo_decision_guide.__doc__)


if __name__ == "__main__":
    # Run ONE at a time — uncomment as you go.
    demo_retry_policy()
    # demo_error_as_state()
    # demo_recursion_limit()
    # demo_debug_state()
    # demo_decision_guide()
