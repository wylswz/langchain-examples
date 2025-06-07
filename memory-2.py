from typing import Literal
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.graph import END, START, Graph
from langgraph.store.memory import InMemoryStore

import uuid
from typing_extensions import TypedDict, Annotated
from operator import add
from langgraph.prebuilt import create_react_agent
from enum import Enum
from utils import remove_think


llm = ChatOllama(model="qwen3:4b")



class Preference(BaseModel):
    key: str = Field(
        alias="key",
        description="偏好设置的 key，例如语气，风格等",
    )
    value: str = Field(
        alias="value", 
        description="偏好设置值"
    )

class Preferences(BaseModel):
    preferences: list[Preference]

class Relationship(BaseModel):
    src: str = Field(description="源实体")
    relationship: str = Field(description="关系")
    tgt: str = Field(description="目标实体")

    def __hash__(self):
        return hash((self.src, self.relationship, self.tgt))

class Relationships(BaseModel):
    relationships: list[Relationship] = Field(alias="relationships", description="relationships between entities")

preference_store: dict = dict()
relationship_store = set()


PROMPT_EXTRACT_PREFERENCE = ChatPromptTemplate.from_template("""
从对话关系中提取用户偏好。
如果对话中没有明确表达偏好的意图，返回空数组

对话如下:
{conversations}
""")

PROMPT_EXTRACT_RELATIONSHIP = ChatPromptTemplate.from_template("""
从对话关系中提取实体之间的关系
一组关系由源实体，关系类型和目标实体组成。
例如
我爸叫李刚，那么关系是 "李刚 是爸爸 我"
class A 继承 class B，那么关系是 "B 父类 A"
Simon还没还我钱，那么关系是 "我 寨主 Simon"

如果对话中没有明确表达关系，返回空数组

在提取实体关系之前，请先进行仔细地思考，尽可能地发掘实体之间隐藏的关系。

对话如下:
{conversations}
""")


preference_extractor = PROMPT_EXTRACT_PREFERENCE | llm.with_structured_output(Preferences)
relationship_extractor = PROMPT_EXTRACT_RELATIONSHIP | llm.with_structured_output(Relationships)

SYSTEM_PROMPT = """
    你是一个带有长期记忆的聊天机器人。你的任务是和用户展开对话。
    用户在对话的过程中会向记忆中添加一些实体和实体之间的关系，以及一些用户偏好。确保你的
    对话风格是严格基于记忆来展开的。
    """

agent = create_react_agent(
    model=llm, 
    tools=[], 
    prompt=SYSTEM_PROMPT,
    debug=True
    )


class State(TypedDict):
    """State for the memory chain."""

    chat_history: Annotated[list[str],..., add]
    background: str

def get_thread_id(c: RunnableConfig):
    return c['configurable']['thread_id']

def bot_chat_step(s: State, config: RunnableConfig, *, store: BaseStore):
    background = """
    老虎约了刘华强在饭店里见，想让刘华强不要再问祥子要钱。
    对话展开的时候，刘华强还不知道老虎叫他来干嘛。
    刘华强带着跃平，大海和一些礼物前往。
    华强和老虎都是石家庄的黑社会，地头蛇。
    """
    q = s['chat_history'][-1]
    message = f"""
    对话背景：
    {background}
    基于之前对话的记忆，继续对话:
    Memory: {store.get((get_thread_id(config),), "memory")}

    User's query:
    {q}
    """
    msg = agent.invoke({
        'messages': [message],
        'conversations': s["chat_history"][-10:]
    })
    return {
        'chat_history': [msg['messages'][-1].content]
    }

def human_step(s: State):
    human_message = interrupt("type input")
    return {
        'chat_history': [human_message]
    }

def evaluate_step(s: State, config: RunnableConfig, *, store: BaseStore):
    relationships = relationship_extractor.invoke({
        'conversations': s["chat_history"][-2:]
    })
    preference = preference_extractor.invoke({
        'conversations': s["chat_history"][-2:]
    })
    for r in relationships.relationships:
        if r not in relationship_store:
            relationship_store.add(r)
    for p in preference.preferences:
        if preference_store.get(p.key) is None:
            preference_store[p.key] = p.value
    preferences = ""
    relationships = ""
    for r in relationship_store:
        relationships += f"{r.src} {r.relationship} {r.tgt}\n"
    for k, v in preference_store.items():
        preferences += f"{k}: {v}\n"
    store.put((get_thread_id(config),), "memory", {
        'preferences': preferences,
        'relationships': relationships,
    })

    

memory = InMemoryStore()
g = Graph()
g.add_edge(START, human_step.__name__)
g.add_node(bot_chat_step)
g.add_node(human_step)
g.add_node(evaluate_step)
g.add_edge(human_step.__name__, bot_chat_step.__name__)
g.add_edge(bot_chat_step.__name__, evaluate_step.__name__)
g.add_edge(evaluate_step.__name__, human_step.__name__)
g = g.compile()


# Test case
"""

欸，强子，刚来啊，等你半天了啊
介绍一下啊，这是我弟弟，枕套

强子，我比你玩儿早点儿，但你强子，名声，我老虎也略知一二。听说你挺讲义气重情份。以后有什么事，说话

痛苦，我就喜欢你这脾气！挑明了说吧，祥子找我了。祥子是我小兄弟儿，做生意赚了点钱。可我手上这帮弟兄们，得靠他养活啊。希望你看在我面子上，从今以后不要再找他

"""