# This is a simulation of a rap battle using multi agent pattern

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, Interrupt
from langgraph.checkpoint.memory import MemorySaver

from typing_extensions import TypedDict
from typing import Annotated

import operator

class State(TypedDict):
    rounds: Annotated[list[tuple], operator.add]
    total_rounds: int
    # winner of each round
    winner: Annotated[list[str], operator.add]

def rapper_1(state: State):
    return Command(
        goto=rapper_2.__name__,
        update={'rounds': [(rapper_1.__name__, 'yo man')]}
    )

def rapper_2(state: State):
    if len(state['rounds']) >= state['total_rounds'] * 2:
        return Command(
            goto=judge.__name__,
            update={}
        )

    return Command(
        goto=rapper_1.__name__,
        update={'rounds': [(rapper_2.__name__, 'you know WTF time it is?')]}
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
    print(chunk)