from checkpointer import graph, MyException
from langgraph.checkpoint.memory import MemorySaver


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
except MyException:
    print("failed for the first time")

try:
    resume()
except MyException:
    print("failed for the second time")

try:
    resume()
except MyException:
    print("failed for the third time")

print(cg.get_state({"configurable": {"thread_id": "thread-1"}}).values)