from typing_extensions import Annotated, TypedDict
from operator import add
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
# This is a demo for subgraph streaming where both parent and child graphs call LLMs
# and we can see messages streamed out in real-time


# Define the state for both parent and child graphs
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add]


# Initialize the LLM model
ollama = ChatOllama(model="qwen3:8b", streaming=True)
openai = ChatOpenAI(model="gpt-4o-mini", streaming=True)

# ============ SUBGRAPH 1 (Child Graph) ============


def subgraph1_node(state: State) -> State:
    """Child graph 1 node that calls LLM"""
    print("\n[SUBGRAPH 1] Calling LLM...")

    ollama.invoke([HumanMessage(content="hello")])

    print("[SUBGRAPH 1] Complete")
    return {}


# Build the first subgraph
def create_subgraph1():
    subgraph = StateGraph(State)
    subgraph.add_node("subgraph1_llm", subgraph1_node)

    subgraph.add_edge(START, "subgraph1_llm")
    subgraph.add_edge("subgraph1_llm", END)

    return subgraph.compile()


# ============ SUBGRAPH 2 (Child Graph) ============


def subgraph2_node(state: State) -> State:
    """Child graph 2 node that calls LLM"""
    print("\n[SUBGRAPH 2] Calling LLM...")

    openai.invoke([HumanMessage(content="hello")])

    print("[SUBGRAPH 2] Complete")
    return {}


# Build the second subgraph
def create_subgraph2():
    subgraph = StateGraph(State)
    subgraph.add_node("subgraph2_llm", subgraph2_node)

    subgraph.add_edge(START, "subgraph2_llm")
    subgraph.add_edge("subgraph2_llm", END)

    return subgraph.compile()


# ============ PARENT GRAPH ============


def parent_node(state: State) -> State:
    """Parent graph node that calls LLM"""
    print("\n[PARENT] Calling LLM...")

    openai.invoke([HumanMessage(content="hello")])

    print("[PARENT] Complete")
    return {}


# Build the parent graph
def create_parent_graph():
    # Create the subgraphs
    subgraph1 = create_subgraph1()
    subgraph2 = create_subgraph2()

    # Create parent graph
    parent = StateGraph(State)
    parent.add_node("parent_llm", parent_node)
    parent.add_node("subgraph1", subgraph1)  # Add subgraph 1 as a node
    parent.add_node("subgraph2", subgraph2)  # Add subgraph 2 as a node

    parent.add_edge(START, "parent_llm")
    # Both subgraphs run in parallel after parent_llm
    parent.add_edge("parent_llm", "subgraph1")
    parent.add_edge("parent_llm", "subgraph2")
    # Both subgraphs go to END
    parent.add_edge("subgraph1", END)
    parent.add_edge("subgraph2", END)

    return parent.compile()


# ============ MAIN EXECUTION ============

if __name__ == "__main__":
    graph = create_parent_graph()

    # Example question
    question = "hello"

    print("=" * 80)
    print(f"USER QUESTION: {question}")
    print("=" * 80)

    initial_state = {"messages": [HumanMessage(content=question)]}

    # Stream the graph execution with messages mode
    print("\n" + "=" * 80)
    print("STREAMING OUTPUT (messages mode):")
    print("=" * 80)

    for ns, chunk in graph.stream(
        initial_state, stream_mode="messages", subgraphs=True
    ):
        # () is parent graph
        # ('subgraph1',) is subgraph 1
        # ('subgraph2',) is subgraph 2
        print(f"{ns}: {chunk[0].content}")
