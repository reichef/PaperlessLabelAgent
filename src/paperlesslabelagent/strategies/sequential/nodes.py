from typing import Any

from paperlesslabelagent.core.nodes.classification import classify_document
from paperlesslabelagent.core.nodes.entities import merge_confirmed_new_entities
from paperlesslabelagent.core.nodes.review import apply_review_answer, check_and_correct_single_proposal, print_proposal
from paperlesslabelagent.core.state import FileProposal
from paperlesslabelagent.strategies.sequential.state import SequentialAgentState
from langgraph.types import interrupt


def propose_for_document(state: SequentialAgentState) -> dict[str, Any]:
    """Step 3 (sequential strategy): classify every not-yet-confirmed file in one pass."""
    print("In Sequential")
    existing_entities = state["existingEntities"]
    proposals: dict[str, FileProposal] = dict(state.get("proposals", {}))
    iteration = state.get("iteration", 0) + 1

    for filename, text in state["file_texts"].items():
        previous_proposal = proposals.get(filename, {})
        if previous_proposal.get("confirmed"):
            continue

        proposals[filename] = classify_document(
            filename,
            text,
            existing_entities,
            rejected_existing_tags=previous_proposal.get("rejected_existing_tags") or [],
            rejected_existing_correspondent=previous_proposal.get("rejected_existing_correspondent"),
            rejected_existing_document_type=previous_proposal.get("rejected_existing_document_type"),
            rejected_new_tags=previous_proposal.get("rejected_new_tags") or [],
            rejected_new_correspondent=previous_proposal.get("rejected_new_correspondent"),
            rejected_new_document_type=previous_proposal.get("rejected_new_document_type"),
        )

    return {"proposals": proposals, "iteration": iteration}


def print_proposals(state: SequentialAgentState) -> dict[str, Any]:
    """Prints every file's proposal to the console (filename + matches only, no document text)."""
    proposals = state.get("proposals", {})

    if not proposals:
        print("No proposals yet.")
        return {}

    for proposal in proposals.items():
        print_proposal(proposal)

    return {}


def check_and_correct_proposals(state: SequentialAgentState) -> dict[str, Any]:
    """Step 4 (sequential strategy): hallucination-check every proposal in one pass."""
    existing_entities = state["existingEntities"]
    proposals: dict[str, FileProposal] = dict(state.get("proposals", {}))

    for filename, proposal in proposals.items():
        proposals[filename] = check_and_correct_single_proposal(filename, proposal, existing_entities)

    return {"proposals": proposals}


def user_verify_proposals(state: SequentialAgentState) -> dict[str, Any]:
    """
    Interacts with the user to determine whether the proposals (assigned entities and new
    entities) provided by the AI fit the user's opinion. For each not-yet-resolved file,
    pauses with the full proposal.
    """
    proposals: dict[str, FileProposal] = dict(state.get("proposals", {}))
    existing_entities = state["existingEntities"]
    confirmed_new_entities = state.get("confirmed_new_entities", [])

    for filename, proposal in proposals.items():
        if proposal.get("confirmed") or proposal.get("needs_retry"):
            continue  # already resolved on a prior pass through the retry loop

        answer = interrupt({"kind": "verify", "filename": filename, "proposal": proposal})
        proposal = apply_review_answer(proposal, answer)
        proposals[filename] = proposal
        if proposal.get("confirmed"):
            existing_entities, confirmed_new_entities = merge_confirmed_new_entities(existing_entities, proposal, confirmed_new_entities)

    return {"proposals": proposals, "existingEntities": existing_entities, "confirmed_new_entities": confirmed_new_entities}
