from langgraph.graph import StateGraph, START, END
from paperlesslabelagent.state import AgentState
from paperlesslabelagent.nodes import *

workflow = StateGraph(AgentState)
workflow.add_node(node="fetch_existing_entities", action=fetch_existing_entities)
workflow.add_node(node="load_documents", action=load_documents)
workflow.add_node(node="propose_for_document", action=propose_for_document)
workflow.add_node(node="print_proposals", action=print_proposals)

workflow.add_edge(START, "fetch_existing_entities")
workflow.add_edge("fetch_existing_entities", "load_documents")
workflow.add_edge("load_documents", "propose_for_document")
workflow.add_edge("propose_for_document", "print_proposals")
workflow.add_edge("print_proposals", END)  # temporary — END moves once review/new-entity nodes exist

graph = workflow.compile()