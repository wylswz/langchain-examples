import sys
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.types import interrupt
from langchain_core.messages import BaseMessage, HumanMessage
from typing_extensions import Annotated, Sequence, TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents import AgentState

model = ChatOpenAI(model="gpt-4o-mini")

@tool
def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@tool
def multiply_two_numbers(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


class State(AgentState):
    pass

def prepare(state: State) -> State:
    print("prepare")
    return {"messages": [state["messages"][0]]}

agent = create_agent(
    tools=[add_two_numbers, multiply_two_numbers], 
    model=model,
    system_prompt="""You are a helpful math agent that helps resolving math problems by calling the tools. 
    Explain your reasoning before calling the tool.
    
    """,
    middleware=[HumanInTheLoopMiddleware(
        interrupt_on={
            "add_two_numbers": {
                "allowed_decisions": ["approve", "reject"],
                "description": "Please review this tool execution"
            }
        }
    )]
)

def cleanup(state: State) -> State:
    print("cleanup")
    return {}

graph = StateGraph(State)
graph.add_node("prepare", prepare)
graph.add_node("agent", agent)
graph.add_node("cleanup", cleanup)
graph.add_edge(START, "prepare")
graph.add_edge("prepare", "agent")
graph.add_edge("agent", "cleanup")
graph.add_edge("cleanup", END)

graph
