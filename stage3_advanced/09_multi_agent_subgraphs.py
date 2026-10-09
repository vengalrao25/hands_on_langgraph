"""
LangGraph 09: Multi-Agent & Subgraphs (compose many graphs into a system)
-------------------------------------------------------------------------
One big agent with 20 tools gets confused and unmaintainable. The production
answer is to SPECIALISE: several small focused agents, coordinated. LangGraph
composes them two ways:

  SUPERVISOR  : a router agent decides which WORKER handles each step, workers
                report back, repeat until done. (Most common multi-agent pattern.)
  SUBGRAPHS   : a whole compiled graph is used as a NODE inside a bigger graph —
                modular, testable units you snap together.

WHY SPECIALISE:
  - each agent has a tight prompt + only its own tools -> better decisions
  - you can test/replace one agent without touching others
  - clear ownership: the "researcher", the "writer", the "reviewer"

THIS FILE:
  1. Supervisor routing to worker agents
  2. A subgraph used as a node in a parent graph
  3. When to go multi-agent (and when NOT to)
"""

from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------------------------------------------------------
# 1. SUPERVISOR — a router agent delegates to workers
# ---------------------------------------------------------------------------
def demo_supervisor():
    """
    The supervisor looks at progress and picks the next worker (or FINISH).
    Workers do their bit and route BACK to the supervisor. The supervisor is the
    'brain' deciding the workflow; workers are the 'hands'. Bounded by recursion_limit.
    """
    class State(TypedDict):
        task: str
        research: str
        draft: str
        next: str

    def supervisor(s: State) -> dict:
        # simple, cheap policy (in practice an LLM call can make this decision)
        if not s.get("research"):
            nxt = "researcher"
        elif not s.get("draft"):
            nxt = "writer"
        else:
            nxt = "FINISH"
        return {"next": nxt}

    def researcher(s: State) -> dict:
        facts = llm.invoke(f"List 3 key facts about: {s['task']}. Be brief.").content
        return {"research": facts}

    def writer(s: State) -> dict:
        post = llm.invoke(f"Write a 3-sentence summary using these facts:\n{s['research']}").content
        return {"draft": post}

    b = StateGraph(State)
    b.add_node("supervisor", supervisor)
    b.add_node("researcher", researcher)
    b.add_node("writer", writer)
    b.add_edge(START, "supervisor")
    b.add_conditional_edges("supervisor", lambda s: s["next"],
                            {"researcher": "researcher", "writer": "writer", "FINISH": END})
    b.add_edge("researcher", "supervisor")   # workers report back
    b.add_edge("writer", "supervisor")
    app = b.compile()

    print("=== supervisor multi-agent ===")
    out = app.invoke({"task": "the benefits of unit testing"}, config={"recursion_limit": 12})
    print("Research:\n", out["research"])
    print("\nDraft:\n", out["draft"])
    print()


# ---------------------------------------------------------------------------
# 2. SUBGRAPH — a compiled graph used as a node
# ---------------------------------------------------------------------------
def demo_subgraph():
    """
    Build a small 'analysis' graph, compile it, and drop it into a parent graph as
    a single node. The subgraph runs as one step from the parent's view. This is
    how you build big systems from small, independently-testable graphs. (Parent
    and subgraph share the state keys they need to pass data.)
    """
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        text: str
        sentiment: str
        summary: str

    # --- subgraph: analyse text (sentiment + summary) ---
    def get_sentiment(s: State) -> dict:
        return {"sentiment": llm.invoke(f"One word sentiment of: {s['text']}").content.strip()}

    def get_summary(s: State) -> dict:
        return {"summary": llm.invoke(f"Summarise in 5 words: {s['text']}").content.strip()}

    sub = StateGraph(State)
    sub.add_node("sentiment", get_sentiment)
    sub.add_node("summary", get_summary)
    sub.add_edge(START, "sentiment")
    sub.add_edge("sentiment", "summary")
    sub.add_edge("summary", END)
    analysis_subgraph = sub.compile()

    # --- parent: respond, using the subgraph as one node ---
    def respond(s: State) -> dict:
        return {"messages": [AIMessage(f"[{s['sentiment']}] {s['summary']}")]}

    parent = StateGraph(State)
    parent.add_node("analyse", analysis_subgraph)   # <- whole graph as a node
    parent.add_node("respond", respond)
    parent.add_edge(START, "analyse")
    parent.add_edge("analyse", "respond")
    parent.add_edge("respond", END)
    app = parent.compile()

    out = app.invoke({
        "text": "The new update is fast and the UI is gorgeous, I love it!",
        "messages": [],
    })
    print("=== subgraph as a node ===")
    print("sentiment:", out["sentiment"])
    print("summary  :", out["summary"])
    print("response :", out["messages"][-1].content)
    print()


# ---------------------------------------------------------------------------
# 3. WHEN TO GO MULTI-AGENT (read this)
# ---------------------------------------------------------------------------
def demo_when_to_use():
    """
    GO MULTI-AGENT WHEN:
      - one agent has too many tools/responsibilities and makes bad choices
      - tasks split into clear specialities (research / code / review / format)
      - you want to test, swap, or scale parts independently

    STAY SINGLE-AGENT WHEN:
      - the task is simple — multi-agent adds latency, cost, and coordination bugs
      - a chain or one create_agent already does the job well

    COSTS TO RESPECT:
      - every hop between agents is more LLM calls -> more $ and latency
      - more autonomy = harder to predict and debug -> lean on tracing (LangSmith)
      - ALWAYS bound coordination loops with recursion_limit

    RULE: start with the simplest thing that works (chain -> single agent ->
    multi-agent). Add agents only when a real limitation forces it.
    """
    print(demo_when_to_use.__doc__)


if __name__ == "__main__":
    # Run ONE at a time — uncomment as you go.
    # demo_supervisor()
    demo_subgraph()
    # demo_when_to_use()
