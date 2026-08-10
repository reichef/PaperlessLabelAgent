from langgraph.types import Command

from paperlesslabelagent.core.nodes.review import collect_hallucination_answer, collect_review_answer
from paperlesslabelagent.core.state import AgentState


def run_interactive(graph, initial_state: AgentState, thread_id: str) -> AgentState:
    """Runs a compiled graph that uses interrupt()-based human review to completion."""

    
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(initial_state, config=config)

    while "__interrupt__" in result:
        interrupt_content = result["__interrupt__"][0].value
        if interrupt_content["kind"] == "hallucination":
            answer = collect_hallucination_answer(interrupt_content)
        else:  # "verify"
            answer = collect_review_answer(interrupt_content)
        result = graph.invoke(Command(resume=answer), config=config)

    return result
