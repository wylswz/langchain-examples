from typing_extensions import Annotated, TypedDict
from operator import add
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

# This is a demo for subgraph streaming where both parent and child graphs call LLMs
# and we can see messages streamed out in real-time

# Define the state for both parent and child graphs
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add]

# Initialize the LLM model
model = ChatOllama(model="qwen3:8b", streaming=True)

# ============ SUBGRAPH (Child Graph) ============

def subgraph_node(state: State) -> State:
    """Child graph node that calls LLM"""
    print("\n[SUBGRAPH] Calling LLM...")
    
    response = model.invoke([HumanMessage(content="hello")])
    
    print(f"[SUBGRAPH] Complete")
    return {
        "messages": [AIMessage(content=f"[Subgraph]: {response.content}")]
    }

# Build the subgraph
def create_subgraph():
    subgraph = StateGraph(State)
    subgraph.add_node("subgraph_llm", subgraph_node)
    
    subgraph.add_edge(START, "subgraph_llm")
    subgraph.add_edge("subgraph_llm", END)
    
    return subgraph.compile()

# ============ PARENT GRAPH ============

def parent_node(state: State) -> State:
    """Parent graph node that calls LLM"""
    print("\n[PARENT] Calling LLM...")
    
    response = model.invoke([HumanMessage(content="hello")])
    
    print(f"[PARENT] Complete")
    return {
        "messages": [AIMessage(content=f"[Parent]: {response.content}")]
    }

# Build the parent graph
def create_parent_graph():
    # Create the subgraph
    subgraph = create_subgraph()
    
    # Create parent graph
    parent = StateGraph(State)
    parent.add_node("parent_llm", parent_node)
    parent.add_node("subgraph", subgraph)  # Add subgraph as a node
    
    parent.add_edge(START, "parent_llm")
    parent.add_edge("parent_llm", "subgraph")
    parent.add_edge("subgraph", END)
    
    return parent.compile()

# ============ MAIN EXECUTION ============

if __name__ == '__main__':
    graph = create_parent_graph()
    
    # Example question
    question = "hello"
    
    print("=" * 80)
    print(f"USER QUESTION: {question}")
    print("=" * 80)
    
    initial_state = {
        "messages": [HumanMessage(content=question)]
    }
    
    # Stream the graph execution with messages mode
    print("\n" + "=" * 80)
    print("STREAMING OUTPUT (messages mode):")
    print("=" * 80)
    
    for chunk in graph.stream(initial_state, stream_mode="messages", subgraphs=True):
        print(chunk)