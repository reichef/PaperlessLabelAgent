from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from paperlesslabelagent.state import AgentState
from paperlesslabelagent.nodes import *


# Total classification attempts allowed
MAX_CLASSIFICATION_ATTEMPTS = 3


def route_after_verification(state: AgentState) -> str:
    """After user verification: if any proposal still needs retry, go re-propose those
    files - unless the retry cap has been hit, in which case stop regardless, leaving
    those files unconfirmed for the user to handle manually."""
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


workflow = StateGraph(AgentState)
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