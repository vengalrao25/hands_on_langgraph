"""
LangGraph 07: Human-in-the-Loop (pause for approval, then resume)
-----------------------------------------------------------------
Some actions are too risky to let an LLM do unsupervised: sending an email,
issuing a refund, running a DB delete, posting publicly. Human-in-the-loop (HITL)
lets the graph PAUSE before such a step, wait for a person to approve/edit/reject,
then CONTINUE exactly where it stopped.

This only works because of persistence (file 06): when the graph interrupts, its
full state is checkpointed. The human can respond seconds or DAYS later; you load
the thread and resume. No persistence -> no HITL.

TWO WAYS TO INTERRUPT:
  - STATIC : compile(interrupt_before=["risky_node"]) — always pause before a node.
  - DYNAMIC: call interrupt(payload) inside a node, resume with Command(resume=...).
We show the static approach (simplest, robust) and explain the dynamic one.

THE RESUME TRICK: to continue a paused graph, invoke it again with None as input
(same thread_id). LangGraph picks up from the saved checkpoint.

THIS FILE:
  1. Pause before a sensitive action (approve / reject)
  2. Inspect the pending action while paused
  3. EDIT the state before resuming (human corrects the plan)
"""

from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class State(TypedDict):
    request: str
    draft_email: str
    sent: bool


def draft(state: State) -> dict:
    email = llm.invoke(f"Write a short email for this request: {state['request']}").content
    return {"draft_email": email}


def send_email(state: State) -> dict:
    # imagine this REALLY sends — that's why we gate it behind a human.
    print(f"   >>> [ACTUALLY SENDING EMAIL]\n   {state['draft_email'][:80]}...")
    return {"sent": True}


def build():
    b = StateGraph(State)
    b.add_node("draft", draft)
    b.add_node("send_email", send_email)
    b.add_edge(START, "draft")
    b.add_edge("draft", "send_email")
    b.add_edge("send_email", END)
    # PAUSE before send_email every time — the human decides whether to proceed.
    return b.compile(checkpointer=MemorySaver(), interrupt_before=["send_email"])


# ---------------------------------------------------------------------------
# 1. APPROVE / REJECT
# ---------------------------------------------------------------------------
def demo_approve_reject():
    """
    Run until the interrupt (draft is done, send is pending). Then a human decides:
    resume (invoke with None) to send, or just don't resume to reject.
    """
    app = build()
    cfg = {"configurable": {"thread_id": "email-1"}}

    print("=== human-in-the-loop: approve/reject ===")
    app.invoke({"request": "Ask the team to submit timesheets by Friday."}, cfg)

    state = app.get_state(cfg)
    print("PAUSED. Next node would be:", state.next)             # ('send_email',)
    print("Draft awaiting approval:\n  ", state.values["draft_email"][:80], "...")

    # --- a human approves -> resume with None ---
    approved = True
    if approved:
        print("\nHuman APPROVED -> resuming...")
        final = app.invoke(None, cfg)        # None = continue from checkpoint
        print("Sent?", final["sent"])
    else:
        print("\nHuman REJECTED -> not resuming. Email never sent.")
    print()


# ---------------------------------------------------------------------------
# 2. EDIT BEFORE RESUMING — human fixes the draft
# ---------------------------------------------------------------------------
def demo_edit_then_resume():
    """
    While paused, the human can CORRECT the state with update_state, then resume.
    The graph continues using the edited values — human oversight that actually
    changes the outcome, not just a yes/no gate.
    """
    app = build()
    cfg = {"configurable": {"thread_id": "email-2"}}

    print("=== human-in-the-loop: edit then resume ===")
    app.invoke({"request": "Tell the client the project is delayed."}, cfg)

    print("Original draft:\n  ", app.get_state(cfg).values["draft_email"][:80], "...")

    # human overrides the draft with their own wording
    app.update_state(cfg, {"draft_email": "Hi, a brief update: we need 3 more days. Thank you for your patience."})
    print("Human edited the draft. Resuming with the edited version...")

    final = app.invoke(None, cfg)
    print("Sent?", final["sent"])
    print()


def demo_dynamic_note():
    """
    DYNAMIC interrupts (newer, more flexible): call interrupt(payload) INSIDE a
    node to pause anywhere and ask the human a question; resume by invoking with
    Command(resume=<answer>). Use it when WHETHER to pause depends on runtime data
    (e.g. only ask for approval on refunds over $100).

        from langgraph.types import interrupt, Command
        def node(state):
            decision = interrupt({"amount": state["amount"]})   # pauses here
            ...
        app.invoke(Command(resume="approve"), config)           # resumes
    """
    print(demo_dynamic_note.__doc__)


if __name__ == "__main__":
    # Run ONE at a time — uncomment as you go.
    demo_approve_reject()
    # demo_edit_then_resume()
    # demo_dynamic_note()
