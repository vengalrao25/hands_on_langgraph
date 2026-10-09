"""
LangGraph 01: Why LangGraph? (and your first StateGraph)
--------------------------------------------------------
You already build pipelines with LCEL (prompt | llm | parser). LCEL is a DAG —
data flows one way, start to finish, no going back. That's perfect for fixed
pipelines and falls apart the moment you need:

  - LOOPS      : "keep refining until it's good enough" (a chain can't go back)
  - BRANCHING  : "if the answer is unsafe, route to a review step"
  - STATE      : a shared, evolving object every step reads/writes
  - PAUSE/RESUME: stop for human approval, then continue exactly where you left off
  - MEMORY     : durable state across calls, per conversation

LangGraph is a STATE MACHINE for LLM apps. You define:
  - a STATE (the shared data),
  - NODES (functions that read state and return updates),
  - EDGES (what runs next — fixed or conditional).
The graph runs nodes, merging their updates into state, following edges, possibly
looping, until it reaches the end.

MENTAL MODEL:
  LCEL chain  = a straight assembly line (DAG).
  LangGraph   = a flowchart with decisions and loops (a graph).

USE A CHAIN when the steps are fixed and one-directional. Reach for LANGGRAPH the
moment you need loops, branching, persistence, or human-in-the-loop. (Note:
create_agent from Stage 5 is itself a prebuilt LangGraph — this folder is you
learning to build that machinery yourself.)

THIS FILE: the smallest possible graph, to lock in the node/edge/state loop.
"""

from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------------------------------------------------------
# THE STATE — the shared object that flows through the graph.
# It's just a TypedDict describing what data the graph carries.
# ---------------------------------------------------------------------------
class State(TypedDict):
    topic: str       # input
    joke: str        # filled by our node


# ---------------------------------------------------------------------------
# A NODE — a function: takes the current state, returns a PARTIAL update.
# You return ONLY the keys you changed; LangGraph merges them into the state.
# ---------------------------------------------------------------------------
def write_joke(state: State) -> dict:
    response = llm.invoke(f"Tell a one-line joke about {state['topic']}.")
    return {"joke": response.content}   # partial update -> merged into state


# ---------------------------------------------------------------------------
# 1. BUILD + RUN A ONE-NODE GRAPH
# ---------------------------------------------------------------------------
def demo_minimal_graph():
    """
    Wiring: START -> write_joke -> END. Compile turns the definition into a
    runnable app. Invoke it with the initial state; get the final state back.
    """
    builder = StateGraph(State)
    builder.add_node("write_joke", write_joke)
    builder.add_edge(START, "write_joke")   # entry
    builder.add_edge("write_joke", END)     # exit
    app = builder.compile()

    result = app.invoke({"topic": "databases"})

    print("=== minimal graph ===")
    print("Final state:", result)            # {'topic': 'databases', 'joke': '...'}
    print("Joke:", result["joke"])
    print()


# ---------------------------------------------------------------------------
# 2. MULTI-NODE — state accumulates as it flows
# ---------------------------------------------------------------------------
def demo_two_nodes():
    """
    Two nodes in sequence, each adding to the SAME state object. Node 2 reads
    what node 1 wrote. This is the LangGraph version of a sequential chain — but
    now every step shares one evolving state (the foundation for loops/branching).
    """
    class S(TypedDict):
        topic: str
        joke: str
        rating: str

    def make_joke(s: S) -> dict:
        return {"joke": llm.invoke(f"One-line joke about {s['topic']}.").content}

    def rate_joke(s: S) -> dict:
        # reads the joke node 1 produced
        return {"rating": llm.invoke(f"Rate this joke 1-10, number only: {s['joke']}").content}

    builder = StateGraph(S)
    builder.add_node("make_joke", make_joke)
    builder.add_node("rate_joke", rate_joke)
    builder.add_edge(START, "make_joke")
    builder.add_edge("make_joke", "rate_joke")   # node1 -> node2
    builder.add_edge("rate_joke", END)
    app = builder.compile()

    result = app.invoke({"topic": "Python"})
    print("=== two nodes (state accumulates) ===")
    print("Joke  :", result["joke"])
    print("Rating:", result["rating"])
    print()


# ---------------------------------------------------------------------------
# 3. INSPECT THE GRAPH STRUCTURE
# ---------------------------------------------------------------------------
def demo_visualise():
    """
    A compiled graph can describe itself — handy for understanding/debugging.
    get_graph().draw_ascii() prints the node/edge structure as text.
    """
    builder = StateGraph(State)
    builder.add_node("write_joke", write_joke)
    builder.add_edge(START, "write_joke")
    builder.add_edge("write_joke", END)
    app = builder.compile()

    print("=== graph structure ===")
    try:
        print(app.get_graph().draw_ascii())
    except Exception:
        # draw_ascii needs an extra dep; fall back to listing nodes/edges
        print("Nodes:", list(app.get_graph().nodes))
        print("Edges:", [(e.source, e.target) for e in app.get_graph().edges])
    print()


if __name__ == "__main__":
    # Run ONE at a time — uncomment as you go.
    # demo_minimal_graph()
    # demo_two_nodes()
    demo_visualise()
