from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import StateGraph, START, END

from paperlesslabelagent.core.nodes.entities import fetch_existing_entities, load_documents
from paperlesslabelagent.core.nodes.resultpersistence import persist_file_proposals, persist_new_entities
from paperlesslabelagent.core.state import ExistingMatch, NewEntityProposal
from paperlesslabelagent.strategies.iterative.nodes import classify_current_document, review_current_proposal, select_next_document
from paperlesslabelagent.strategies.iterative.state import IterativeAgentState

# ExistingMatch/NewEntityProposal aren't on langgraph's built-in msgpack allowlist, so
# checkpointing them (this graph interrupts on essentially every document, so it checkpoints
# constantly) would otherwise emit a "deserializing unregistered type" warning that
# langgraph says will become a hard error in a future version. Register them explicitly.
CHECKPOINT_SERDE = JsonPlusSerializer(allowed_msgpack_modules=[ExistingMatch, NewEntityProposal])


def route_after_persistence(state: IterativeAgentState) -> str:
    """Iteratively performs document classification, user-review and persistence per document.
    Uses accepted entities (tags, document_types and correspondents) in classification of next documents to avoid different syntax for same semantics.
    This avoids a merging step for entities before persistence.
    Each confirmed document is persisted (and moved out of the input folder) before the next one
    is classified, so aborting the run neither loses confirmed work nor makes a restart redo it.
    """
    return END if select_next_document(state) is None else "classify_current_document"

workflow = StateGraph(IterativeAgentState)
workflow.add_node(node="fetch_existing_entities", action=fetch_existing_entities)
workflow.add_node(node="load_documents", action=load_documents)
workflow.add_node(node="classify_current_document", action=classify_current_document)
workflow.add_node(node="review_current_proposal", action=review_current_proposal)
workflow.add_node(node="persist_new_entities", action=persist_new_entities)
workflow.add_node(node="persist_file_proposals", action=persist_file_proposals)

workflow.add_edge(START, "fetch_existing_entities")
workflow.add_edge("fetch_existing_entities", "load_documents")
workflow.add_edge("load_documents", "classify_current_document")
workflow.add_edge("classify_current_document", "review_current_proposal")
# Persist after every review, not only after the last document. On a retry (not confirmed) both
# persist nodes are no-ops: persist_file_proposals skips unconfirmed proposals and
# persist_new_entities only finds placeholder ids that confirmation added.
workflow.add_edge("review_current_proposal", "persist_new_entities")
workflow.add_edge("persist_new_entities", "persist_file_proposals")
workflow.add_conditional_edges("persist_file_proposals", route_after_persistence,
    {"classify_current_document": "classify_current_document", END: END},
)

graph = workflow.compile(checkpointer=InMemorySaver(serde=CHECKPOINT_SERDE))
