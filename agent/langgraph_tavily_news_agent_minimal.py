# A demo langGraph agent

"""
LangGraph + Tavily news agent (minimal)

Requirements:
  pip install -U langgraph langchain-openai langchain-tavily typing_extensions

Env vars:
  export OPENAI_API_KEY=...
  export TAVILY_API_KEY=...

Run:
  python langgraph_tavily_news_agent_minimal.py
"""
from __future__ import annotations

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage

# Use the new Tavily integration package (not the deprecated community tool)
from langchain_tavily import TavilySearch


# 1) Define the graph state (messages with an append reducer)
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# 2) Tools — Tavily search tuned for recent news
#    You can adjust max_results/time_range/topic as you like.
tavily_search = TavilySearch(
    max_results=5,
    topic="news",             # news/general/finance
    time_range="week",        # day | week | month | year
    search_depth="advanced",  # basic | advanced
    include_answer=True,       # include a concise answer field
)
TOOLS = [tavily_search]


# 3) LLM — any tools-capable chat model via LangChain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM_PROMPT = (
    "You are a helpful research assistant.\n"
    "When the user asks about current events or 'latest' info, use the tavily_search tool.\n"
    "Always return a concise answer with source URLs when possible."
)


# 4) Agent node: call the model once; model will decide to call tools

def agent(state: State) -> State:
    msgs = state["messages"]
    # Inject system guidance once at start
    if not any(isinstance(m, SystemMessage) for m in msgs):
        msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs
    ai_msg = llm_with_tools.invoke(msgs)
    return {"messages": [ai_msg]}


# 5) Build the graph: START -> agent -> (maybe tools) -> agent -> END
builder = StateGraph(State)

builder.add_node("agent", agent)
# ToolNode executes any tool calls the model requested
builder.add_node("tools", ToolNode(TOOLS))

builder.add_edge(START, "agent")
# Route to the tool node if the model requested tools; otherwise end
builder.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "tools", "action": "tools", "__end__": END}  # map both "tools" and older "action" labels; "__end__" means stop,
)  # tools_condition returns either "tools"/"action" or "__end__"; map accordingly

# After tools run, go back to the agent to produce the final answer
builder.add_edge("tools", "agent")

# (Optional) Memory so your graph keeps history across calls
memory = MemorySaver()
app = builder.compile(checkpointer=memory)


# 6) Demo
if __name__ == "__main__"
    # A thread_id lets the checkpointer persist/recall this conversation
    cfg = {"configurable": {"thread_id": "news-demo"}}

    user_q = (
        "Give me the latest Bitcoin market headlines from this week and cite sources."
    )

    result = app.invoke({"messages": [HumanMessage(content=user_q)]}, config=cfg)

    # Print the last model reply
    print("\n=== ASSISTANT ===\n")
    print(result["messages"][-1].content)
