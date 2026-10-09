"""
LangGraph 08: Streaming (show progress AND tokens from a running graph)
-----------------------------------------------------------------------
A graph can take many steps (loops, tools, multiple LLM calls). Users shouldn't
stare at a blank screen. LangGraph streams at different granularities via
stream_mode:

  stream_mode="updates"  -> after each NODE, yields {node_name: state_update}
                            (great for "Searching... / Generating..." progress)
  stream_mode="values"   -> after each step, yields the FULL state so far
  stream_mode="messages" -> yields LLM tokens as they're generated, with metadata
                            (this is token-by-token streaming inside a graph)

You can also pass a LIST of modes to get several at once.

THIS FILE:
  1. updates — watch which node ran and what it changed
  2. values — see the whole state evolve
  3. messages — stream LLM tokens live from inside the graph
  4. async streaming (astream) — what a server uses
"""

import asyncio
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    outline: str


def make_outline(s: State) -> dict:
    return {"outline": llm.invoke(f"3-bullet outline for a post about {s['topic']}.").content}


def write_post(s: State) -> dict:
    return {"messages": [llm.invoke(f"Write a short post from this outline:\n{s['outline']}")]}


def build():
    b = StateGraph(State)
    b.add_node("make_outline", make_outline)
    b.add_node("write_post", write_post)
    b.add_edge(START, "make_outline")
    b.add_edge("make_outline", "write_post")
    b.add_edge("write_post", END)
    return b.compile()


# ---------------------------------------------------------------------------
# 1. UPDATES — progress, node by node
# ---------------------------------------------------------------------------
def demo_updates():
    """Each yield tells you which node just finished and what it produced."""
    app = build()
    print("=== stream_mode='updates' ===")
    for chunk in app.stream({"topic": "vector databases", "messages": []}, stream_mode="updates"):
        for node, update in chunk.items():
            changed = list(update.keys())
            print(f"  node '{node}' updated: {changed}")
    print()


# ---------------------------------------------------------------------------
# 2. VALUES — the full state after each step
# ---------------------------------------------------------------------------
def demo_values():
    """Useful when a UI re-renders from the complete current state each tick."""
    app = build()
    print("=== stream_mode='values' ===")
    for state in app.stream({"topic": "caching", "messages": []}, stream_mode="values"):
        print(f"  state keys present: {[k for k, v in state.items() if v]}")
    print()


# ---------------------------------------------------------------------------
# 3. MESSAGES — token-by-token from inside the graph
# ---------------------------------------------------------------------------
def demo_messages():
    """
    Stream the actual LLM tokens as nodes generate them. Each yield is
    (message_chunk, metadata); metadata['langgraph_node'] tells you WHICH node the
    token came from — so a UI can stream the right part to the right place.
    """
    app = build()
    print("=== stream_mode='messages' (live tokens) ===")
    for token, meta in app.stream({"topic": "LangGraph", "messages": []}, stream_mode="messages"):
        if token.content:
            print(token.content, end="", flush=True)
    print("\n")


# ---------------------------------------------------------------------------
# 4. ASYNC STREAMING — server-side
# ---------------------------------------------------------------------------
async def demo_astream():
    """astream is the async twin — what a FastAPI endpoint uses to push tokens."""
    app = build()
    print("=== astream (async) ===")
    async for token, meta in app.astream({"topic": "async", "messages": []}, stream_mode="messages"):
        if token.content:
            print(token.content, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    # Run ONE at a time — uncomment as you go.
    # demo_updates()
    # demo_values()
    # demo_messages()
    asyncio.run(demo_astream())
