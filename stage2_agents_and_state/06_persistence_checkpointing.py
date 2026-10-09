"""
LangGraph 06: Persistence & Checkpointing (durable state, time travel, threads)
-------------------------------------------------------------------------------
This is LangGraph's killer feature. Compile a graph with a CHECKPOINTER and it
saves the ENTIRE state after every step, keyed by a thread_id. That single change
unlocks:

  - MEMORY      : same thread_id resumes exactly where it left off (Stage 3 used this)
  - DURABILITY  : crash/restart mid-run -> resume from the last checkpoint
  - INSPECTION  : read the current state any time (get_state)
  - TIME TRAVEL : list every past checkpoint and resume from an OLD one (get_state_history)
  - HUMAN-IN-LOOP: pause, persist, come back later (next file)

A "thread" = one independent run/conversation. thread_id is how you separate users
and how you reconnect to an existing run.

THIS FILE:
  1. Checkpointer = memory across calls
  2. get_state — inspect current state + what runs next
  3. Threads are isolated
  4. Time travel — get_state_history and resume from a past checkpoint
"""

from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chat(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


def build(checkpointer):
    b = StateGraph(State)
    b.add_node("chat", chat)
    b.add_edge(START, "chat"); b.add_edge("chat", END)
    return b.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 1. CHECKPOINTER = MEMORY ACROSS CALLS
# ---------------------------------------------------------------------------
def demo_memory():
    """
    Two separate invokes on the same thread_id. The checkpointer persisted the
    first turn's state, so the second turn sees it. No manual history list —
    persistence IS the memory.
    """
    app = build(MemorySaver())
    cfg = {"configurable": {"thread_id": "user-1"}}

    print("=== persistence = memory ===")
    r1 = app.invoke({"messages": [HumanMessage("My name is Vengal.")]}, cfg)
    print("AI:", r1["messages"][-1].content)
    r2 = app.invoke({"messages": [HumanMessage("What's my name?")]}, cfg)
    print("AI:", r2["messages"][-1].content)   # remembers
    print()


# ---------------------------------------------------------------------------
# 2. get_state — inspect the saved state
# ---------------------------------------------------------------------------
def demo_get_state():
    """
    get_state(config) returns a snapshot: the current values, plus .next (which
    node would run next — empty tuple means the run finished). This is how you
    debug and how UIs show "where are we?".
    """
    app = build(MemorySaver())
    cfg = {"configurable": {"thread_id": "inspect-me"}}
    app.invoke({"messages": [HumanMessage("Remember the number 42.")]}, cfg)

    snap = app.get_state(cfg)
    print("=== get_state ===")
    print("\n\n\n\n\n\n",snap,"\n\n\n\n\n\n")
    print("messages in state:", len(snap.values["messages"]))
    print("next node to run :", snap.next or "(none — finished)")
    print()


# ---------------------------------------------------------------------------
# 3. THREADS ARE ISOLATED
# ---------------------------------------------------------------------------
def demo_thread_isolation():
    """
    Different thread_id = totally separate state. This is multi-user safety: each
    user/conversation gets its own thread; they never see each other's data.
    """
    app = build(MemorySaver())
    app.invoke({"messages": [HumanMessage("I love Python.")]}, {"configurable": {"thread_id": "A"}})

    print("=== thread isolation ===")
    a = app.invoke({"messages": [HumanMessage("What do I love?")]}, {"configurable": {"thread_id": "A"}})
    b = app.invoke({"messages": [HumanMessage("What do I love?")]}, {"configurable": {"thread_id": "B"}})
    print("Thread A (knows):", a["messages"][-1].content)
    print("Thread B (fresh):", b["messages"][-1].content)
    print()


# ---------------------------------------------------------------------------
# 4. TIME TRAVEL — replay/branch from a past checkpoint
# ---------------------------------------------------------------------------
def demo_time_travel():
    """
    get_state_history lists EVERY checkpoint for a thread (newest first). Each has
    a checkpoint_id; pass one back in the config to RESUME from that exact past
    point — to debug, or to branch an alternate path ('what if we'd answered
    differently here?'). Almost no other framework gives you this for free.
    """
    app = build(MemorySaver())
    cfg = {"configurable": {"thread_id": "tt"}}
    app.invoke({"messages": [HumanMessage("Step one.")]}, cfg)
    app.invoke({"messages": [HumanMessage("Step two.")]}, cfg)
    app.invoke({"messages": [HumanMessage("Step three.")]}, cfg)

    history = list[Any](app.get_state_history(cfg))
    print("=== time travel ===")
    print(f"Total checkpoints saved: {len(history)}")
    for i, snap in enumerate(history):
        n = len(snap.values.get("messages", []))
        print(f"  checkpoint[{i}] messages={n} next={snap.next or '(end)'}")
    print("\nResume from an earlier checkpoint by passing its checkpoint_id in config:")
    print('  cfg = {"configurable": {"thread_id": "tt", "checkpoint_id": <id>}}')
    print()


if __name__ == "__main__":
    # Run ONE at a time — uncomment as you go.
    # demo_memory()
    # demo_get_state()
    # demo_thread_isolation()
    demo_time_travel()
