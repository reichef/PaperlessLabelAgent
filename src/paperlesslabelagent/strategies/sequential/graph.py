from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

from paperlesslabelagent.core.nodes.entities import fetch_existing_entities, load_documents
from paperlesslabelagent.strategies.sequential.nodes import (
    check_and_correct_proposals,
    print_proposals,
    propose_for_document,
    user_verify_proposals,
)
from paperlesslabelagent.strategies.sequential.state import SequentialAgentState

MAX_CLASSIFICATION_ATTEMPTS = 3


def route_after_verification(state: SequentialAgentState) -> str:
    """After user verification: if any proposal still needs retry, re-classify those
    files. Stop when MAX_CLASSIFICATION_ATTEMPTS are hit"""
    proposals = state.get("proposals", {})
    still_needs_retry = [filename for filename, proposal in proposals.items() if proposal.get("needs_retry")]

    if not still_needs_retry:
        return END

    if state.get("iteration", 0) >= MAX_CLASSIFICATION_ATTEMPTS:
        print(
            f"\nReached the retry limit ({MAX_CLASSIFICATION_ATTEMPTS}), "
            f"leaving {len(still_needs_retry)} file(s) unconfirmed: {', '.join(still_needs_retry)}"
        )
        return END

    return "propose_for_document"

workflow = StateGraph(SequentialAgentState)
workflow.add_node(node="fetch_existing_entities", action=fetch_existing_entities)
workflow.add_node(node="load_documents", action=load_documents)
workflow.add_node(node="propose_for_document", action=propose_for_document)
workflow.add_node(node="print_proposals", action=print_proposals)
workflow.add_node(node="check_and_correct_proposals", action=check_and_correct_proposals)
workflow.add_node(node="user_verify_proposals", action=user_verify_proposals)

workflow.add_edge(START, "fetch_existing_entities")
workflow.add_edge("fetch_existing_entities", "load_documents")
workflow.add_edge("load_documents", "propose_for_document")
workflow.add_edge("propose_for_document", "print_proposals")
workflow.add_edge("print_proposals", "check_and_correct_proposals")
workflow.add_edge("check_and_correct_proposals", "user_verify_proposals")
workflow.add_conditional_edges("user_verify_proposals", route_after_verification,
    {"propose_for_document": "propose_for_document", END: END},
)

graph = workflow.compile(checkpointer=InMemorySaver())
