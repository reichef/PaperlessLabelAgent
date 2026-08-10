from typing import Any

from langgraph.types import interrupt

from paperlesslabelagent.core.nodes.classification import classify_document
from paperlesslabelagent.core.nodes.entities import merge_confirmed_new_entities
from paperlesslabelagent.core.nodes.review import apply_review_answer, check_and_correct_single_proposal
from paperlesslabelagent.strategies.iterative.state import IterativeAgentState, IterativeFileProposal

# Per-document classification attempts allowed before giving up on that one document and
# moving on (unlike the sequential strategy's MAX_CLASSIFICATION_ATTEMPTS, this cap is
# per-document, not global - see strategies.iterative.state.IterativeFileProposal).
MAX_ATTEMPTS_PER_DOCUMENT = 3


def select_next_document(state: IterativeAgentState) -> str | None:
    """Returns the first filename in file_texts that isn't confirmed and hasn't exceeded
    MAX_ATTEMPTS_PER_DOCUMENT attempts, or None if nothing is left to process.
    """
    proposals = state.get("proposals", {})
    for filename in state.get("file_texts", {}):
        proposal = proposals.get(filename, {})
        if proposal.get("confirmed"):
            continue
        if proposal.get("attempts", 0) >= MAX_ATTEMPTS_PER_DOCUMENT:
            continue
        return filename
    return None


def classify_current_document(state: IterativeAgentState) -> dict[str, Any]:
    """Classifies the next eligible document."""

    print("In Iterative")
    filename = select_next_document(state)
    if filename is None:
        return {"current_filename": None}  # nothing left; route_after_review will end the run

    proposals: dict[str, IterativeFileProposal] = dict(state.get("proposals", {}))
    previous_proposal = proposals.get(filename, {})

    proposal = classify_document(
        filename,
        state["file_texts"][filename],
        state["existingEntities"],
        rejected_existing_tags=previous_proposal.get("rejected_existing_tags") or [],
        rejected_existing_correspondent=previous_proposal.get("rejected_existing_correspondent"),
        rejected_existing_document_type=previous_proposal.get("rejected_existing_document_type"),
        rejected_new_tags=previous_proposal.get("rejected_new_tags") or [],
        rejected_new_correspondent=previous_proposal.get("rejected_new_correspondent"),
        rejected_new_document_type=previous_proposal.get("rejected_new_document_type"),
    )
    # classify_document doesn't know about "attempts" (it's exclusive to this strategy), so
    # it has to be injected onto the dict it returns rather than being part of it already.
    proposal["attempts"] = previous_proposal.get("attempts", 0) + 1
    proposals[filename] = proposal

    return {"proposals": proposals, "current_filename": filename}


def review_current_proposal(state: IterativeAgentState) -> dict[str, Any]:
    """Hallucination-checks and reviews the document classify_current_document just built,
    then - if confirmed - folds any newly-accepted entities into the existingEntities pool
    so later documents can match against them directly. All interrupt-driven or pure (no
    LLM calls), safe to combine in one node for the same reason sequential's
    check_and_correct_proposals/user_verify_proposals are each safe on their own: multiple
    interrupt() calls plus pure logic around them replay safely, only LLM calls don't.
    """
    filename = state.get("current_filename")
    if filename is None:
        return {}  # 
    
    proposals: dict[str, IterativeFileProposal] = dict(state.get("proposals", {}))
    proposal = check_and_correct_single_proposal(filename, proposals[filename], state["existingEntities"])

    if not proposal.get("needs_retry"):  # mirrors user_verify_proposals' skip condition exactly
        answer = interrupt({"kind": "verify", "filename": filename, "proposal": proposal})
        proposal = apply_review_answer(proposal, answer)

    proposals[filename] = proposal

    existing_entities = state["existingEntities"]
    confirmed_new_entities = state.get("confirmed_new_entities", [])
    if proposal.get("confirmed"):
        existing_entities, confirmed_new_entities = merge_confirmed_new_entities(existing_entities, proposal, confirmed_new_entities)

    return {"proposals": proposals, "existingEntities": existing_entities, "confirmed_new_entities": confirmed_new_entities}
