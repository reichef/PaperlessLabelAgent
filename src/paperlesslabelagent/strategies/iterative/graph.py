from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

from paperlesslabelagent.core.nodes.entities import fetch_existing_entities, load_documents
from paperlesslabelagent.strategies.iterative.nodes import classify_current_document, review_current_proposal, select_next_document
from paperlesslabelagent.strategies.iterative.state import IterativeAgentState


def route_after_review(state: IterativeAgentState) -> str:
    """Iteratively performs document classification and user-review per document. 
    Uses accepted entities (tags, document_types and correspondents) in classification of next documents to avoid different syntax for same semantics. 
    This avoids a merging step for entities before persistence. 
    """
    return END if select_next_document(state) is None else "classify_current_document"

workflow = StateGraph(IterativeAgentState)
workflow.add_node(node="fetch_existing_entities", action=fetch_existing_entities)
workflow.add_node(node="load_documents", action=load_documents)
workflow.add_node(node="classify_current_document", action=classify_current_document)
workflow.add_node(node="review_current_proposal", action=review_current_proposal)

workflow.add_edge(START, "fetch_existing_entities")
workflow.add_edge("fetch_existing_entities", "load_documents")
workflow.add_edge("load_documents", "classify_current_document")
workflow.add_edge("classify_current_document", "review_current_proposal")
workflow.add_conditional_edges("review_current_proposal", route_after_review,
    {"classify_current_document": "classify_current_document", END: END},
)

graph = workflow.compile(checkpointer=InMemorySaver())
