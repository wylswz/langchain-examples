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
from langgraph.prebuilt import create_react_agent
from langchain_core.documents import Document
from typing_extensions import TypedDict
from typing import Annotated
from collections import Counter
from utils import remove_think
import operator

from common import get_vector_store

LANG_MODEL = 'deepseek-r1:8b'

vector_store = get_vector_store(collection_name="library").as_retriever()
llm = ChatOllama(model = LANG_MODEL)

@tool(parse_docstring=True)
def retrieve_lyrics(query: str) -> list[Document]:
    """retrieve documents from library

    Args:
        query: query string
    """
    return vector_store.invoke(query)
# llm = llm.bind_tools([retrieve_documents])

rapper_1_background = "A real gangster rapper, a thug, a bitch nigga, sell drugs, fuck bitches, burn cars and drop bombs."
rapper_2_background = "A rap lover, primary job is a police officer."
background = """
you are in an underground rap battle, you will diss each other as hard as you can.
don't rap more than 20 sentences each round.
better flow, higher marks.
mixing different styles and beats is a big plus. 
"""


class State(TypedDict):
    rounds: Annotated[list[tuple], operator.add]
    total_rounds: int
    # winner of each round
    winner: Annotated[list[str], operator.add]

class JudgeResult(TypedDict):
    winner_rapper_name: str

def get_prompt(state: State):
    if state['rounds']:
        return f"""
        you are now responding to {state['rounds'][-1]}, rap back that ass
        and let the motherfucker know who is the real bitch thug. 
        """
    return "Please start rap"

def get_references(background, prompt) -> list[Document]:
    docs: list[Document] = vector_store.invoke(background) + vector_store.invoke(prompt)
    '\n'.join(map(lambda d: d.page_content, docs))

RAPPER_PROMPT = """
{context}

{messages}

you can refer to following lyrics in this battle
<references>
{references}
</references>

Your final answer shoud be included in tag <answer> and </answer>
"""


JUDGE_PROMPT = """
I am going to judge which rapper wins this round
===
{rapper_1_name}'s lyrics:
{rapper_1_lyrics}
===
{rapper_2_name}'s lyrics:
{rapper_2_lyrics}
===

"""

agent_1 = create_react_agent(
    model=llm, 
    tools=[retrieve_lyrics], 
    prompt=PromptTemplate.from_template(f"{background}, you are {rapper_2_background}. your opponent is a {rapper_1_background}"))

agent_2 = create_react_agent(
    model=llm,
    tools=[retrieve_lyrics],
    prompt=PromptTemplate.from_template(f"{background}, you are {rapper_2_background}. your opponent is a {rapper_1_background}")
)

judge_llm = PromptTemplate.from_template(JUDGE_PROMPT) | llm.with_structured_output(JudgeResult)
rapper_llm = PromptTemplate.from_template(RAPPER_PROMPT) | llm

def rapper_1(state: State):
    ctx = f"{background}, you are {rapper_1_background}, and your opponent is {rapper_2_background}"
    res = rapper_llm.invoke({
        'context': f"{background}, you are {rapper_1_background}, and your opponent is {rapper_2_background}",
        'messages': get_prompt(state),
        'references': get_references(ctx, get_prompt(state))
    })
    return Command(
        goto=rapper_2.__name__,
        update={'rounds': [(rapper_1.__name__, remove_think(res.content))]}
    )
        
        

def rapper_2(state: State):
    if len(state['rounds']) >= state['total_rounds'] * 2:
        return Command(
            goto=judge.__name__,
        )
    ctx = f"{background}, you are {rapper_2_background}. your opponent is a {rapper_1_background}"
    res = rapper_llm.invoke({
        'context': ctx,
        'messages': get_prompt(state),
        'references': get_references(ctx, get_prompt(state))
    })
    return Command(
        goto=rapper_1.__name__,
        update={'rounds': [(rapper_2.__name__, remove_think(res.content))]}
    )
        
def judge(state: State):
    check_round = len(state['winner'])
    if check_round >= state['total_rounds']:
        return Command(goto=END)
    i = check_round * 2
    rapper_1_lyrics = state['rounds'][i]
    rapper_2_lyrics = state['rounds'][i + 1]

    res: JudgeResult = judge_llm.invoke({
        'rapper_1_name': rapper_1.__name__,
        'rapper_2_name': rapper_2.__name__,
        'rapper_1_lyrics': rapper_1_lyrics,
        'rapper_2_lyrics': rapper_2_lyrics
    })
    
    return Command(
        goto=judge.__name__,
        update={'winner': [res['winner_rapper_name']]}
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


for (chunk, meta) in battle.stream({'total_rounds': 2}, config=config, stream_mode="messages"):
    thinking = False
    if isinstance(chunk, AIMessageChunk):
        print(chunk.content, end='')