from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware
from langchain.agents import AgentState
from langchain.agents.middleware.context_editing import ContextEdit, TokenCounter
from langchain.tools import tool
from langchain_ollama import ChatOllama

from langgraph.graph.state import StateGraph
from langgraph.graph import START, END
from langchain_core.messages import AnyMessage, HumanMessage

# This is an example of doing context editing in a langchain agent
# it propagates the context from the parent graph to the agent by creating an inner middleware


class State(AgentState):
    ctx: str


@tool
def book_ticket(username: str):
    """Book a ticket for the user"""
    return {
        "success": True,
        "ticket_id": "1234567890",
        "message": f"Ticket successfully booked for {username}",
    }


@tool
def get_ticket_status(ticket_id: str):
    """Get the status of a ticket"""
    return {
        "departure": "Beijing",
        "arrival": "Shanghai",
        "date": "2025-11-10",
        "time": "10:00",
        "price": 100,
        "status": "booked",
    }


def init_ctx(state: State) -> State:
    return {"ctx": "The user is called yunlu"}


def call_agent(state: State) -> State:
    class InjectCtxEdit(ContextEdit):
        def apply(
            self, messages: list[AnyMessage], *, count_tokens: TokenCounter
        ) -> None:
            print(f"Injecting ctx: {state['ctx']}")
            messages.append(HumanMessage(content=state["ctx"]))

    agent = create_agent(
        tools=[book_ticket, get_ticket_status],
        model=ChatOllama(model="qwen3:8b"),
        system_prompt="""
        You are a agent that helps user by calling tools. If there is no more tools to be called,
        finish the conversation.
        """,
        middleware=[
            ContextEditingMiddleware(
                edits=[
                    InjectCtxEdit(),
                ]
            )
        ],
    )
    return agent.invoke({"messages": state["messages"]})


graph = StateGraph(State)
graph.add_node("init_ctx", init_ctx)
graph.add_node("call_agent", call_agent)
graph.add_edge(START, "init_ctx")
graph.add_edge("init_ctx", "call_agent")
graph.add_edge("call_agent", END)
