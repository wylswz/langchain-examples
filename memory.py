# this is a demo of memory

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict


class State(TypedDict):
    cnt: int



def step_1(s: State, config, *, store: BaseStore):
    ns = (config['configurable']['user_id'], 'g1')
    v = store.get(ns, 'k1')
    if not v:
        store.put(ns, 'k1', {'cnt': 1})
    else:
        store.put(ns, 'k1', {'cnt': v.value['cnt'] + 1})
    return {'cnt': s['cnt'] + 1}

def step_2(s: State, config, *, store: BaseStore):
    ns = (config['configurable']['user_id'], 'g1')
    v = store.get(ns, 'k1')
    store.put(ns, 'k1', {'cnt': v.value['cnt'] + 1})
    return {'cnt': s['cnt'] + 1}

g = StateGraph(State)

g.add_node(step_1).add_node(step_2).add_edge(START, step_1.__name__).add_edge(step_1.__name__, step_2.__name__).add_edge(step_2.__name__, END)

checkpointer = InMemorySaver()
memory = InMemoryStore()
wf = g.compile(checkpointer=checkpointer, store=memory)

config = {
    'thread_id': '1',
    'user_id': 'u1'
}

for val in wf.stream({'cnt': 0}, config=config):
    print(val)
for val in wf.stream({'cnt': 0}, config=config):
    print(val)

# we have cnt: 2 in state
# while cnt: 4 in memory
print(memory.get(('u1', 'g1'), 'k1').value)