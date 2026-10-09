"""
LangGraph 02: State & Reducers (how updates MERGE — the concept people miss)
---------------------------------------------------------------------------
A node returns a partial update; LangGraph merges it into the state. The QUESTION
is: HOW does it merge? That's what a REDUCER decides, per field.

  DEFAULT reducer = OVERWRITE. New value replaces the old.
      state['answer'] = 'A'  -> node returns {'answer': 'B'}  -> state['answer'] = 'B'

  CUSTOM reducer = COMBINE. e.g. APPEND to a list instead of replacing it.
      Annotated[list, add]          -> new list is concatenated onto the old
      Annotated[list, add_messages] -> messages are appended (and de-duped by id)

This matters enormously: in a chat/agent you want messages to ACCUMULATE, not get
overwritten each turn. The add_messages reducer is why MessagesState "remembers"
within a run — each node's messages get appended, not replaced.

THIS FILE:
  1. Default overwrite behaviour
  2. A reducer that appends to a list
  3. add_messages — the reducer behind every chat/agent graph
  4. Multiple fields, different reducers
"""

from email import message
import operator
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()


# ---------------------------------------------------------------------------
# 1. DEFAULT = OVERWRITE
# ---------------------------------------------------------------------------
def demo_overwrite():
    """
    No reducer annotation -> each node's value REPLACES the previous. Two nodes
    both writing 'value' -> only the last one survives.
    """
    class State(TypedDict):
        value: str

    def node_a(s): return {"value": "from A"}
    def node_b(s): return {"value": "from B"}   # overwrites A

    b = StateGraph(State)
    b.add_node("a", node_a); b.add_node("b", node_b)
    b.add_edge(START, "a"); b.add_edge("a", "b"); b.add_edge("b", END)
    app = b.compile()

    print("=== default (overwrite) ===")
    print(app.invoke({"value": "initial"}))   # {'value': 'from B'}
    print()


# ---------------------------------------------------------------------------
# 2. REDUCER = APPEND (accumulate instead of replace)
# ---------------------------------------------------------------------------
def demo_append_reducer():
    """
    Annotated[list, operator.add] tells LangGraph: when a node returns this field,
    CONCATENATE it onto the existing list instead of replacing. Now BOTH nodes'
    contributions survive. This is how you accumulate logs, results, steps.
    """
    class State(TypedDict):
        steps: Annotated[list, operator.add]   # reducer = list concatenation

    def node_a(s): return {"steps": ["did A"]}
    def node_b(s): return {"steps": ["did B"]}

    b = StateGraph(State)
    b.add_node("a", node_a); b.add_node("b", node_b)
    b.add_edge(START, "a"); b.add_edge("a", "b"); b.add_edge("b", END)
    app = b.compile()

    print("=== reducer (append) ===")
    print(app.invoke({"steps": []}))   # {'steps': ['did A', 'did B']}
    print()


# ---------------------------------------------------------------------------
# 3. add_messages — the reducer behind chat/agent state
# ---------------------------------------------------------------------------
def demo_add_messages():
    """
    add_messages is a smart list reducer for messages: it APPENDS new messages,
    and if a message has the same id as an existing one, it UPDATES it instead of
    duplicating. This is exactly what conversation/agent state needs. MessagesState
    is just {'messages': Annotated[list, add_messages]} under the hood.
    """
    class ChatState(TypedDict):
        messages: Annotated[list, add_messages]

    def respond(s: ChatState) -> dict:
        # return a NEW message; the reducer appends it to the running list
        return {"messages": [AIMessage("Hi! How can I help?")]}

    b = StateGraph(ChatState)
    b.add_node("respond", respond)
    b.add_edge(START, "respond"); b.add_edge("respond", END)
    app = b.compile()

    result = app.invoke({"messages": [HumanMessage("Hello")]})
    print("=== add_messages reducer ===")
    for m in result["messages"]:
        print(f"  {m.type}: {m.content}")   # human: Hello / ai: Hi! ...
    print("(the human message was kept AND the ai message appended)")
    print()


# ---------------------------------------------------------------------------
# 4. MULTIPLE FIELDS, DIFFERENT REDUCERS
# ---------------------------------------------------------------------------
def demo_mixed_state():
    """
    Real state mixes reducer types: messages append, a counter sums, a 'final
    answer' overwrites. You pick the right merge behaviour PER FIELD — that
    precision is what makes graph state powerful.
    """
    class State(TypedDict):
        messages: Annotated[list, add_messages]   # append
        tokens_used: Annotated[int, operator.add] # sum
        status: str                               # overwrite

    def step1(s): return {"messages": [AIMessage("working...")], "tokens_used": 10, "status": "running"}
    def step2(s): return {"messages": [AIMessage("done")], "tokens_used": 15, "status": "complete"}

    b = StateGraph(State)
    b.add_node("s1", step1); b.add_node("s2", step2)
    b.add_edge(START, "s1"); b.add_edge("s1", "s2"); b.add_edge("s2", END)
    app = b.compile()

    out = app.invoke({"messages": [], "tokens_used": 0, "status": "init"})
    print("=== mixed reducers ===")
    print("messages   :", len(out["messages"]), "(appended)")
    print("\n\n\n\n\n ", out["messages"] , "\n\n\n")
    print("tokens_used:", out["tokens_used"], "(summed: 10+15)")
    print("status     :", out["status"], "(overwritten: last wins)")
    print()


if __name__ == "__main__":
    # Run ONE at a time — uncomment as you go.
    # demo_overwrite()
    # demo_append_reducer()
    # demo_add_messages()
    demo_mixed_state()
