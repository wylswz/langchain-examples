from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents import AgentState

# This is a demo for invoking agent in a langgraph node and using human in the loop middleware to interrupt the graph execution

model = ChatOpenAI(model="gpt-4o-mini")


@tool
def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@tool
def multiply_two_numbers(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


@tool
def subtract_two_numbers(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b


@tool
def divide_two_numbers(a: int, b: int) -> int:
    """Divide two numbers"""
    return a / b


class State(AgentState):
    pass


def prepare(state: State) -> State:
    print("prepare")
    return {"messages": [state["messages"][0]]}


def execute(state: State) -> State:
    agent = create_agent(
        tools=[
            add_two_numbers,
            multiply_two_numbers,
            subtract_two_numbers,
            divide_two_numbers,
        ],
        model=model,
        system_prompt="""You are a helpful math agent that helps resolving math problems by calling the tools. 
        Explain your reasoning before calling the tool.
        """,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "add_two_numbers": {
                        "allowed_decisions": ["approve", "reject"],
                        "description": "Please review this tool execution",
                    },
                    "multiply_two_numbers": {
                        "allowed_decisions": ["approve", "reject"],
                        "description": "Please review this tool execution",
                    },
                    "subtract_two_numbers": {
                        "allowed_decisions": ["approve", "reject"],
                        "description": "Please review this tool execution",
                    },
                    "divide_two_numbers": {
                        "allowed_decisions": ["approve", "reject"],
                        "description": "Please review this tool execution",
                    },
                }
            )
        ],
    )
    res = agent.invoke({"messages": state["messages"]})
    return {"messages": [res["messages"][-1]]}


def cleanup(state: State) -> State:
    print("cleanup")
    return {}


graph = StateGraph(State)
graph.add_node("prepare", prepare)
graph.add_node("execute", execute)
graph.add_node("cleanup", cleanup)
graph.add_edge(START, "prepare")
graph.add_edge("prepare", "execute")
graph.add_edge("execute", "cleanup")
graph.add_edge("cleanup", END)

graph
