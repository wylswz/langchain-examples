from langgraph.config import get_stream_writer
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph
from langgraph.graph import START, END

# This is a demo of using custom stream mode to control token streaming.
# The front-end doesn't output the token until a control signal is received.

SIG_BEGIN_STREAM = "begin_stream"

def get_model():
    return ChatOllama(model="llama3.2:3b", temperature=0.0)

prompt = "show me some latest ai agent trends"

class State(TypedDict):
    messages: list[str]


def prepare(s: State) -> State:
    get_model().invoke(prompt)
    return {"messages": []}

def begin_stream(s: State) -> State:
    get_stream_writer()({"signal": SIG_BEGIN_STREAM})
    return {"messages": []}

def stream(s: State) -> State:
    get_model().invoke(prompt)
    return {"messages": []}

graph = StateGraph(State)
graph.add_node("prepare", prepare)
graph.add_node("begin_stream", begin_stream)
graph.add_node("stream", stream)
graph.add_edge(START, "prepare")
graph.add_edge("prepare", "begin_stream")
graph.add_edge("begin_stream", "stream")
graph.add_edge("stream", END)
graph = graph.compile()


if __name__ == '__main__':
    thread_id = '1'
    print_token = False
    for (stream_type, chunk) in graph.stream({"messages": ["Hello, how are you?"]}, stream_mode=["custom", "messages"]):
        if stream_type == "messages":
            if print_token:
                print(chunk[0].content, end='')
        elif stream_type == "custom":
            if chunk["signal"] == SIG_BEGIN_STREAM:
                print("begin_stream")
                print_token = True
