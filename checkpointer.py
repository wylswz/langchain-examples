from langchain_core.messages import BaseMessage
from langgraph.types import Command
from typing_extensions import Annotated, Sequence, TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
import random
from operator import add

class MyException(Exception):
    pass



def add_messages(messages: Sequence[BaseMessage], new_messages: Sequence[BaseMessage] | BaseMessage) -> Sequence[BaseMessage]:
    if isinstance(new_messages, BaseMessage):
        messages.append(new_messages)
        return messages
    return messages + new_messages


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    failures: Annotated[int, add]

def random_fail() -> State:
    if random.random() < 0.5:
        raise MyException("Random failure")

def step_1(state: State) -> State:
    return {"messages": [HumanMessage(content="this is step 1")]}


def step_2(state: State) -> State:
    random_fail()
    return {"messages": [HumanMessage(content="this is step 2")]}


def step_3(state: State) -> State:
    return {"messages": [HumanMessage(content="this is step 3")]}


graph = StateGraph(State)
graph.add_node("step_1", step_1)
graph.add_node("step_2", step_2)
graph.add_node("step_3", step_3)
graph.add_edge(START, "step_1")
graph.add_edge("step_1", "step_2")
graph.add_edge("step_2", "step_3")
graph.add_edge("step_3", END)

saver = MemorySaver()
cg = graph.compile(checkpointer=saver)

def run():
    stream = cg.stream({"messages": []}, {"configurable": {"thread_id": "thread-1"}})
    for event in stream:
        print(event)

def resume():
    stream = cg.stream(None, {"configurable": {"thread_id": "thread-1"}})
    for event in stream:
        print(event)

try:
    run()
except MyException as e:
    print("failed for the first time")

try:
    resume()
except MyException as e:
    print("failed for the second time")

try:
    resume()
except MyException as e:
    print("failed for the third time")

print(cg.get_state({"configurable": {"thread_id": "thread-1"}}).values)