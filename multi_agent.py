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
from utils import remove_think, read_lines
import operator

from common import get_vector_store, get_img_summaries_llm, img_to_txt

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

rapper_1_background = """
A real gangster rapper, a thug, a bitch nigga, sell drugs, fuck bitches, burn cars and drop bombs. Outlawz grown on the streets, always
hate the police.
"""
rapper_2_background = "A rap lover, primary job is a police officer."


background = """
you are in an underground rap battle, you will diss each other as hard as you can.
dissing appearance is also encouraged.
better flow, higher marks.
mixing different styles and beats is a big plus. 
"""


class State(TypedDict):
    rounds: Annotated[list[tuple], operator.add]
    total_rounds: int
    # winner of each round
    winner: Annotated[list[str], operator.add]
    play_as: str

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

you can try multiple times to improve your rap, any your final version shoud be included in tag <answer> and </answer>.
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

show how you score each rapper, and append a single line at the end which contains winner's name
"""

rapper_1_appearance = img_to_txt('assets/rappers/eazy.jpg')
rapper_2_appearance = img_to_txt('assets/rappers/police.jpg')

print("rapper 1 is comming: ")
print(rapper_1_appearance)

print("rapper 2 is comming: ")
print(rapper_2_appearance)


judge_llm = PromptTemplate.from_template(JUDGE_PROMPT) | llm
rapper_llm = PromptTemplate.from_template(RAPPER_PROMPT) | llm

def rapper_1(state: State):
    if state['play_as'] == rapper_1.__name__:
        lyrics = read_lines('rap the ass')
    else:
        ctx = f"{background}, you are {rapper_1_background}, and your opponent is {rapper_2_background}, he's like {rapper_2_appearance}"
        print("\nrapper_1's round")
        res = rapper_llm.invoke({
            'context': f"{background}, you are {rapper_1_background}, and your opponent is {rapper_2_background}",
            'messages': get_prompt(state),
            'references': get_references(ctx, get_prompt(state))
        })
        lyrics = remove_think(res.content)
    return Command(
        goto=rapper_2.__name__,
        update={'rounds': [(rapper_1.__name__, lyrics)]}
    )
        
        

def rapper_2(state: State):
    if len(state['rounds']) >= state['total_rounds'] * 2:
        return Command(
            goto=judge.__name__,
        )
    if state['play_as'] == rapper_2.__name__:
        lyrics = read_lines('rap the ass')
    else:
        ctx = f"{background}, you are {rapper_2_background}. your opponent is a {rapper_1_background}, he's like {rapper_1_appearance}"
        print("\nrapper_2's round")
        res = rapper_llm.invoke({
            'context': ctx,
            'messages': get_prompt(state),
            'references': get_references(ctx, get_prompt(state))
        })
        lyrics = remove_think(res.content)
    return Command(
        goto=rapper_1.__name__,
        update={'rounds': [(rapper_2.__name__, lyrics)]}
    )
        
def judge(state: State):
    check_round = len(state['winner'])
    if check_round >= state['total_rounds']:
        return Command(goto=END)
    i = check_round * 2
    rapper_1_lyrics = state['rounds'][i]
    rapper_2_lyrics = state['rounds'][i + 1]

    res = judge_llm.invoke({
        'rapper_1_name': rapper_1.__name__,
        'rapper_2_name': rapper_2.__name__,
        'rapper_1_lyrics': rapper_1_lyrics,
        'rapper_2_lyrics': rapper_2_lyrics
    })
    
    
    return Command(
        goto=judge.__name__,
        update={'winner': [res.content.splitlines()[-1]]}
    )


g = StateGraph(State)
g.add_edge(START, rapper_1.__name__)
g.add_node(rapper_1)
g.add_node(rapper_2)
g.add_node(judge)

battle = g.compile(checkpointer=MemorySaver())
print(battle.get_graph().draw_ascii())
config = {
    'thread_id': '1',
    'recursion_limit': 100,
}


for (chunk, meta) in battle.stream({'total_rounds': 2, 'play_as': rapper_1.__name__}, config=config, stream_mode="messages"):
    thinking = False
    if isinstance(chunk, AIMessageChunk):
        print(chunk.content, end='')