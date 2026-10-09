"""
LangGraph 05: Build a Tool-Calling Agent FROM SCRATCH (demystify create_agent)
------------------------------------------------------------------------------
In Stage 5 you used create_agent — a black box. Here you build the SAME thing with
a graph, so the ReAct loop is never magic again. An agent is just a cycle:

    agent (LLM) --has tool calls?--> tools --> back to agent
                --no tool calls?--> END

  - the AGENT node calls the LLM (which may request tool calls)
  - a conditional edge checks: did it ask for tools?
        yes -> run the TOOLS node, loop back to agent with the results
        no  -> we have a final answer -> END

That loop — think, act, observe, repeat — IS an agent. create_agent just wires
this for you. LangGraph gives two prebuilt helpers so you don't hand-roll the
plumbing:
    ToolNode        — runs whatever tools the last AI message requested
    tools_condition — the router: "tools" if there are tool calls, else END

THIS FILE:
  1. Define tools + state
  2. The agent node (LLM bound to tools)
  3. Wire the cycle with ToolNode + tools_condition
  4. Run it and watch the loop
"""

from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return {"paris": "18°C, rain", "chennai": "34°C, sunny"}.get(city.lower(), "unknown")


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression like '3 * (4 + 2)'."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))   # sandboxed-ish for demo
    except Exception as e:
        return f"error: {e}"


tools = [get_weather, calculate]
llm_with_tools = llm.bind_tools(tools)


# ---------------------------------------------------------------------------
# State — just a messages list with the add_messages reducer
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# The agent node: call the LLM; it returns either an answer or tool requests
# ---------------------------------------------------------------------------
def agent_node(state: AgentState) -> dict:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# ---------------------------------------------------------------------------
# 1. BUILD THE AGENT GRAPH (the cycle)
# ---------------------------------------------------------------------------
def build_agent():
    b = StateGraph(AgentState)
    b.add_node("agent", agent_node)
    b.add_node("tools", ToolNode(tools))     # runs requested tools, appends ToolMessages

    b.add_edge(START, "agent")
    # tools_condition routes to "tools" if the last AI msg has tool_calls, else END
    b.add_conditional_edges("agent", tools_condition)
    b.add_edge("tools", "agent")             # after tools, loop BACK to the agent
    return b.compile()


# ---------------------------------------------------------------------------
# 2. RUN IT
# ---------------------------------------------------------------------------
def demo_agent():
    """
    Watch the cycle: agent asks for a tool -> tools run -> back to agent -> agent
    has the result -> final answer -> END. You built the ReAct loop yourself.
    """
    app = build_agent()
    result = app.invoke(
        {"messages": [HumanMessage("What's the weather in Paris, and what is 15 * 23?")]},
        config={"recursion_limit": 10},
    )
    print("=== agent from scratch ===")
    for m in result["messages"]:
        m.pretty_print()
    print()


# ---------------------------------------------------------------------------
# 3. SEE THE LOOP STEP BY STEP
# ---------------------------------------------------------------------------
def demo_watch_loop():
    """
    Stream the graph to see each node fire in order — agent, tools, agent — proving
    it really is a cycle, not a straight line.
    """
    app = build_agent()
    print("=== watch the loop (node by node) ===")
    for chunk in app.stream(
        {"messages": [HumanMessage("Weather in Chennai?")]},
        config={"recursion_limit": 10},
        stream_mode="updates",   # yields {node_name: update} after each node
    ):
        for node, update in chunk.items():
            last = update["messages"][-1]
            kind = "tool_call" if getattr(last, "tool_calls", None) else last.type
            print(f"  node='{node}' -> {kind}: {str(last.content)[:50]}")
    print()


if __name__ == "__main__":
    # Run ONE at a time — uncomment as you go.
    demo_agent()
    # demo_watch_loop()
