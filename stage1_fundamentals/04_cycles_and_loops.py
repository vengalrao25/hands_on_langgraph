"""
LangGraph 04: Cycles & Loops (the superpower a chain literally cannot do)
-------------------------------------------------------------------------
An LCEL chain is a DAG — it can never go backwards. But tons of real LLM patterns
are LOOPS: "generate -> critique -> if not good enough, regenerate -> repeat".
LangGraph does this naturally: a conditional edge can point BACK to an earlier
node, creating a cycle, and you loop until a condition says stop.

    generate -> evaluate --(not good enough)--> generate   (loop back!)
                         --(good enough)------> END

THE TWO THINGS A LOOP NEEDS:
  1. a conditional edge that can route back to an earlier node
  2. a STOP condition (and a safety cap) so it can't loop forever

SAFETY: every loop MUST be bounded. Track an attempt counter in state AND rely on
recursion_limit as a backstop, or a bad evaluator will loop until it runs out of
money. Unbounded loops are the #1 LangGraph footgun.

THIS FILE:
  1. A self-correcting generate/evaluate loop
  2. The attempt counter + max-attempts stop
  3. recursion_limit as the hard backstop
"""

from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# ---------------------------------------------------------------------------
# 1. SELF-CORRECTING LOOP — generate, evaluate, retry until good
# ---------------------------------------------------------------------------
def demo_refine_loop():
    """
    Write a tagline, have a (strict) judge score it, and if it scores low, loop
    back and try again — up to a max. This 'reflection' pattern measurably
    improves quality and is impossible in a one-shot chain.
    """
    class State(TypedDict):
        topic: str
        draft: str
        score: int
        attempts: int

    def generate(s: State) -> dict:
        prompt = f"Write a punchy 6-word marketing tagline for: {s['topic']}."
        if s.get("draft"):
            prompt += f" The previous attempt was weak: '{s['draft']}'. Make it better."
        return {"draft": llm.invoke(prompt).content.strip(), "attempts": s.get("attempts", 0) + 1}

    def evaluate(s: State) -> dict:
        raw = llm.invoke(
            f"Score this tagline 1-10 for punchiness. Reply with ONLY the number.\n{s['draft']}"
        ).content.strip()
        score = int("".join(c for c in raw if c.isdigit()) or 0)
        return {"score": score}

    def should_continue(s: State) -> str:
        # stop if good enough OR we've tried too many times
        if s["score"] >= 8 or s["attempts"] >= 3:
            return "stop"
        return "retry"

    b = StateGraph(State)
    b.add_node("generate", generate)
    b.add_node("evaluate", evaluate)
    b.add_edge(START, "generate")
    b.add_edge("generate", "evaluate")
    b.add_conditional_edges("evaluate", should_continue, {"retry": "generate", "stop": END})
    app = b.compile()

    print("=== self-correcting loop ===")
    out = app.invoke({"topic": "a productivity app for developers"})
    print(f"Final tagline (after {out['attempts']} attempt(s), score {out['score']}):")
    print(" ", out["draft"])
    print()


# ---------------------------------------------------------------------------
# 2. RECURSION LIMIT — the hard backstop against infinite loops
# ---------------------------------------------------------------------------
def demo_recursion_limit():
    """
    Suppose the stop condition is buggy and never fires. recursion_limit caps the
    TOTAL number of steps; exceeding it raises GraphRecursionError instead of
    looping forever. ALWAYS set this for any graph with a cycle — it's your seatbelt.
    """
    from langgraph.errors import GraphRecursionError

    class State(TypedDict):
        count: int

    def increment(s: State) -> dict:
        return {"count": s["count"] + 1}

    def loop_forever(s: State) -> str:
        return "again"   # BUG: never returns "stop"

    b = StateGraph(State)
    b.add_node("increment", increment)
    b.add_edge(START, "increment")
    b.add_conditional_edges("increment", loop_forever, {"again": "increment", "stop": END})
    app = b.compile()

    print("=== recursion limit (infinite loop, caught) ===")
    try:
        app.invoke({"count": 0}, config={"recursion_limit": 5})
    except GraphRecursionError:
        print("GraphRecursionError raised at the limit — loop stopped safely, no runaway bill.")
    print()


if __name__ == "__main__":
    # Run ONE at a time — uncomment as you go.
    demo_refine_loop()
    # demo_recursion_limit()
