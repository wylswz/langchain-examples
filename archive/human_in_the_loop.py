# This is a demo that illustrates how to implement human-in-the-loop pattern
# it simply requests an approve of a proposal
# you can approve the proposal by typing yes or true in stdin


from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, Interrupt
from langgraph.checkpoint.memory import MemorySaver


from typing_extensions import TypedDict
from typing import Annotated

class State(TypedDict):
    proposal: str
    proposal_approved: bool


def propose_step(state: State):
    return {
        "proposal": "this is a proposal"
    }


def human_approve_step(state: State):
    answer = interrupt(state["proposal"])
    return {
        "proposal_approved": answer in {'yes', 'YES', 'true', 'TRUE', 'y', 'Y'}
    }


def approved(state: State):
    if state["proposal_approved"]:
        return END
    return propose_step.__name__


g = StateGraph(State)
g.add_node(propose_step)
g.add_node(human_approve_step)
g.add_edge(START, propose_step.__name__)
g.add_edge(propose_step.__name__, human_approve_step.__name__)

g.add_conditional_edges(human_approve_step.__name__, approved, [END, propose_step.__name__])

# app = g.compile(checkpointer=MemorySaver())


# config = {'thread_id': '1'}

# ended = False
# stream = app.stream({
#         "foo": "bar"
#     }, config=config)
# while not ended:
#     if not stream:
#         break
#     interrupted = False
#     for t in stream:
#         if '__interrupt__' in t:
#             approved = input(t['__interrupt__'])
#             stream = app.invoke(Command(resume=approved), config=config, stream_mode="updates")
#             interrupted = True
#         else:
#             print(t)

#     if not interrupted:
#         # current stream is consumed without interruption
#         # so we can end the loop
#         stream = []    