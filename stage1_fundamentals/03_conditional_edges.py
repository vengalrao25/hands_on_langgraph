"""
LangGraph 03: Conditional Edges (branching — the thing chains can't do well)
----------------------------------------------------------------------------
So far edges were FIXED: A always goes to B. Conditional edges let the graph
DECIDE the next node at runtime based on the current state. This is routing /
branching — the first real superpower over linear chains.

    add_conditional_edges(source, router_fn, path_map)
      - router_fn(state) -> returns a KEY (a string)
      - path_map         -> {key: next_node_name}

Think of router_fn as the "if/elif/else" of your graph. It inspects state and
returns which branch to take.

PRODUCTION USES:
  - classify the request, route to the right specialist node
  - safety/guardrail check -> route unsafe inputs to a refusal node
  - "do we have enough info?" -> route to ask-clarifying-question vs answer

THIS FILE:
  1. A router that branches on a classification
  2. The full pattern with an LLM classifier
  3. A guardrail branch (route + refuse)
"""

from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------------------------------------------------------
# 1. BASIC BRANCHING — router returns a key, path_map picks the node
# ---------------------------------------------------------------------------
def demo_basic_branch():
    """
    The classifier node sets state['category']; the router reads it and returns
    a key; path_map sends execution to the matching node. Note there is NO edge
    from classify directly to a handler — the conditional edge IS the wiring.
    """
    class State(TypedDict):
        question: str
        category: str
        answer: str

    def classify(s: State) -> dict:
        q = s["question"].lower()
        cat = "math" if any(c.isdigit() for c in q) else "general"
        return {"category": cat}

    def route(s: State) -> str:        # <- the decision function
        return s["category"]           # returns "math" or "general"

    def handle_math(s): return {"answer": llm.invoke(f"Solve: {s['question']}").content}
    def handle_general(s): return {"answer": llm.invoke(s["question"]).content}

    b = StateGraph(State)
    b.add_node("classify", classify)
    b.add_node("math", handle_math)
    b.add_node("general", handle_general)
    b.add_edge(START, "classify")
    b.add_conditional_edges("classify", route, {"math": "math", "general": "general"})
    b.add_edge("math", END)
    b.add_edge("general", END)
    app = b.compile()

    print("=== basic branching ===")
    for q in ["What is 12 * 9?", "Who wrote Hamlet?"]:
        out = app.invoke({"question": q})
        print(f"[{out['category']:7}] {q} -> {out['answer'][:60]}")
    print()


# ---------------------------------------------------------------------------
# 2. LLM-DRIVEN ROUTING — let the model classify, then branch
# ---------------------------------------------------------------------------
def demo_llm_router():
    """
    A more realistic router: an LLM classifies the intent, the graph routes to a
    specialised handler. This is the 'intent router' pattern behind many assistants
    (billing vs technical vs sales, each with its own prompt/tools).
    """
    class State(TypedDict):
        query: str
        intent: str
        response: str

    def detect_intent(s: State) -> dict:
        intent = llm.invoke(
            f"Classify this query as exactly one word - 'billing', 'technical', or 'other':\n{s['query']}"
        ).content.strip().lower()
        intent = intent if intent in {"billing", "technical"} else "other"
        return {"intent": intent}

    def billing(s): return {"response": "[Billing team] " + llm.invoke(f"Answer this billing query briefly: {s['query']}").content}
    def technical(s): return {"response": "[Tech team] " + llm.invoke(f"Answer this technical query briefly: {s['query']}").content}
    def other(s): return {"response": "[General] " + llm.invoke(s["query"]).content}

    b = StateGraph(State)
    for name, fn in [("detect", detect_intent), ("billing", billing), ("technical", technical), ("other", other)]:
        b.add_node(name, fn)
    b.add_edge(START, "detect")
    b.add_conditional_edges("detect", lambda s: s["intent"],
                            {"billing": "billing", "technical": "technical", "other": "other"})
    for h in ["billing", "technical", "other"]:
        b.add_edge(h, END)
    app = b.compile()

    print("=== LLM router ===")
    for q in ["Why was I charged twice?", "The app crashes on login.", "What's the weather?"]:
        out = app.invoke({"query": q})
        print(f"[{out['intent']:9}] {out['response'][:70]}")
    print()


# ---------------------------------------------------------------------------
# 3. GUARDRAIL BRANCH — route unsafe input away from the expensive path
# ---------------------------------------------------------------------------
def demo_guardrail_branch():
    """
    A cheap safety check first; if it fails, branch straight to a refusal and skip
    the real work. This is the graph version of the is_relevant guardrail you like —
    routing, not just flagging. Saves cost and enforces policy at the edge.
    """
    class State(TypedDict):
        text: str
        safe: bool
        result: str

    def check(s: State) -> dict:
        verdict = llm.invoke(
            f"Is this request safe and on-topic for a coding assistant? Answer yes or no:\n{s['text']}"
        ).content.strip().lower()
        return {"safe": verdict.startswith("y")}

    def proceed(s): return {"result": llm.invoke(s["text"]).content}
    def refuse(s): return {"result": "I can't help with that request."}

    b = StateGraph(State)
    b.add_node("check", check); b.add_node("proceed", proceed); b.add_node("refuse", refuse)
    b.add_edge(START, "check")
    b.add_conditional_edges("check", lambda s: "proceed" if s["safe"] else "refuse",
                            {"proceed": "proceed", "refuse": "refuse"})
    b.add_edge("proceed", END); b.add_edge("refuse", END)
    app = b.compile()

    print("=== guardrail branch ===")
    for t in ["Write a Python function to reverse a string.", "Help me hack my neighbour's wifi."]:
        out = app.invoke({"text": t})
        print(f"[safe={out['safe']}] {out['result'][:60]}")
    print()


if __name__ == "__main__":
    # Run ONE at a time — uncomment as you go.
    demo_basic_branch()
    # demo_llm_router()
    # demo_guardrail_branch()
