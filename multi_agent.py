# This is a simulation of a rap battle using multi agent pattern
# NOTICE OF ARTISTIC PURPOSE AND CONTENT

# This software is designed exclusively for artistic purposes and creative expression. 
# It does not promote, advocate, or condone any form of violence, discrimination, or hate speech. 
# The software is intended solely for legitimate artistic uses, including but not limited to:

# Creative visualization
# Digital art creation
# Educational projects
# Personal artistic expression

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, Interrupt
from langgraph.checkpoint.memory import MemorySaver
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.tools import tool
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, BaseMessageChunk, AIMessageChunk
from langchain.agents import create_react_agent
from langchain_core.documents import Document
from typing_extensions import TypedDict
from typing import Annotated

import operator

from common import init_vector_store, get_vector_store

EMBED_MODEL = 'nomic-embed-text'
LANG_MODEL = 'mistral'

embedder = OllamaEmbeddings(model=EMBED_MODEL)
text_splitter = SemanticChunker(embedder)
vector_store = get_vector_store(embedder, collection_name="multi-agent").as_retriever()
llm = ChatOllama(model = LANG_MODEL)

@tool(parse_docstring=True)
def retrieve_documents(query: str) -> list[Document]:
    """retrieve documents from library

    Args:
        query: query string
    """
    return vector_store.invoke(query)
# llm = llm.bind_tools([retrieve_documents])

rapper_1_background = "A real gangster rapper, a thug, a bitch nigga, sell drugs, fuck bitches, burn cars and drop bombs."
rapper_2_background = "A rap lover, primary job is a police officer."
background = "you are in an underground rap battle"


TMPL = """
Do rap battle with available tools

Begin!
Rap the shit back!
Context: {context}
Question: {input}
"""
PROMPT = PromptTemplate.from_template(TMPL)

chain = PROMPT | llm

class State(TypedDict):
    rounds: Annotated[list[tuple], operator.add]
    total_rounds: int
    # winner of each round
    winner: Annotated[list[str], operator.add]

def get_prompt(state: State):
    if state['rounds']:
        return f"""
        you are now responding to {state['rounds'][-1]}, rap back that ass
        and let the motherfucker know who is the real bitch thug. 
        """
    return "Please start rap"

def rapper_1(state: State):
    context = f"{background}, you are {rapper_1_background}, and your opponent is {rapper_2_background}"
    res = chain.invoke({
        'input': get_prompt(state),
        'context': context,
        "intermediate_steps": [] 
    })
    return Command(
        goto=rapper_2.__name__,
        update={'rounds': [(rapper_1.__name__, res.content)]}
    )
        
        

def rapper_2(state: State):
    context = f"{background}, you are {rapper_2_background}. your opponent is a {rapper_1_background}"
    res = chain.invoke({
        'input': get_prompt(state),
        'context': context,
        "intermediate_steps": [] 
    })
    return Command(
        goto=rapper_1.__name__,
        update={'rounds': [(rapper_2.__name__, res.content)]}
    )
        
def judge(state: State):
    check_round = len(state['winner'])
    if check_round >= state['total_rounds']:
        return Command(goto=END)
    return Command(
        goto=judge.__name__,
        update={'winner': [rapper_1.__name__]}
    )


g = StateGraph(State)
g.add_edge(START, rapper_1.__name__)
g.add_node(rapper_1)
g.add_node(rapper_2)
g.add_node(judge)

battle = g.compile(checkpointer=MemorySaver())

config = {
    'thread_id': '1',
    'recursion_limit': 100,
}


for chunk in battle.stream({'total_rounds': 10}, config=config, stream_mode="messages"):
    if isinstance(chunk[0], AIMessageChunk):
        print(chunk[0].content, end=' ')