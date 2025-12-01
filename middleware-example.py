from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, AgentMiddleware
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain.agents import AgentState

from langgraph.runtime import Runtime

from typing import Any


llm = ChatOllama(model="qwen3:8b")

tools = [TavilySearch(max_results=1)]

class AuditMiddleware(AgentMiddleware):
    def __init__(self):
        pass
    
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(state)
        return None

prompt = """
You are a personal assistant that helps user to accomplish their tasks by doing 
online searches and calling tools.
<instructions>
- do searches actively
- call tools when necessary
</instructions>
"""
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=prompt,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "tavily_search": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "Please review the search results"
                }
            }
        ),
        AuditMiddleware()
    ]
)

