from typing import Any

from langgraph.types import interrupt

from paperlesslabelagent.core.state import FileProposal


def print_proposal(proposal):
    print(f"\n --- Proposition of assignments of tags, correspondent and document types === {proposal[0]} === as well as new ones tags, correspondents and document types ---")

    if proposal[1]["proposed_existing_tags"]:
        for tag in proposal[1]["proposed_existing_tags"]:
            print(f'  Tag: "{tag["name"]}" (confidence={tag["confidence"]:.2f})')
    else:
        print("  Tags: (no match)")

    correspondent = proposal[1]["proposed_existing_correspondent"]
    if correspondent:
        print(f'  Correspondent: "{correspondent["name"]}" (confidence={correspondent["confidence"]:.2f})')
    else:
        print("  Correspondent: (no match)")

    document_type = proposal[1]["proposed_existing_document_type"]
    if document_type:
        print(f'  Document type: "{document_type["name"]}" (confidence={document_type["confidence"]:.2f})')
    else:
        print("  Document type: (no match)")

    for new_tag in proposal[1]["proposed_new_tags"] or []:
        print(f'  New tag proposed: "{new_tag["name"]}" - {new_tag["description"]}')

    new_correspondent = proposal[1]["proposed_new_correspondent"]
    if new_correspondent:
        print(f'  New correspondent proposed: "{new_correspondent["name"]}" - {new_correspondent["description"]}')
    else:
        print("  New correspondent: (no match)")

    new_document_type = proposal[1]["proposed_new_document_type"]
    if new_document_type:
        print(f'  New document type proposed: "{new_document_type["name"]}" - {new_document_type["description"]}')
    else:
        print("  New document type: (no match)")


def ask_yes_no(question: str) -> bool:
    """Prompts on stdin for an explicit y/n answer, reprompting on invalid/empty input."""
    while True:
        raw = input(f"{question} [y/n]: ").strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")


def check_and_correct_single_proposal(filename: str, proposal: FileProposal, existing_entities: dict[str, Any]) -> FileProposal:
    """Checks a single proposal for erroneous assignments of hallucinated existing entries.

    For every tag/correspondent/document_type the LLM matched that doesn't actually exist
    in existing_entities (name not found), and asks whether to
    keep it as a new-entity proposal instead, or reject it and retry.
    """

    tags = existing_entities.get("tags", {}).get("results", [])
    correspondents = existing_entities.get("correspondents", {}).get("results", [])
    document_types = existing_entities.get("document_types", {}).get("results", [])

    tag_names = {item["name"] for item in tags}
    correspondent_names = {item["name"] for item in correspondents}
    document_type_names = {item["name"] for item in document_types}

    kept_tags = []
    for tag in proposal["proposed_existing_tags"]:
        if tag["name"] in tag_names:
            kept_tags.append(tag)
            continue

        # Not a existing tag - the LLM hallucinated it. Use it as a new proposed entity instead of existing one.
        keep_as_new = interrupt({"kind": "hallucination", "entity_type": "tag", "filename": filename, "entity": tag})
        new_tag_proposal = {"entity_type": "tag", "name": tag["name"], "description": tag["reasoning"], "reasoning": tag["reasoning"]}
        if keep_as_new:
            new_tags = proposal.get("proposed_new_tags") or []
            new_tags.append(new_tag_proposal)
            proposal["proposed_new_tags"] = new_tags
        else:
            rejected_new_tags = proposal.get("rejected_new_tags") or []
            rejected_new_tags.append(new_tag_proposal)
            proposal["rejected_new_tags"] = rejected_new_tags
            proposal["needs_retry"] = True
    proposal["proposed_existing_tags"] = kept_tags

    for proposed_existing_entity_type, entity_type, new_entity_type, rejected_new_entity_type, names in (
        ("proposed_existing_correspondent", "correspondent", "proposed_new_correspondent", "rejected_new_correspondent", correspondent_names),
        ("proposed_existing_document_type", "document_type", "proposed_new_document_type", "rejected_new_document_type", document_type_names),
    ):
        value = proposal[proposed_existing_entity_type]
        if value is None or value["name"] in names:
            continue

        # Not a existing correspondent or document_type, hallucinated by LLM --- see comment above.
        keep_as_new = interrupt({"kind": "hallucination", "entity_type": entity_type, "filename": filename, "entity": value})
        new_entity_proposal = {
            "entity_type": entity_type,
            "name": value["name"],
            "description": value["reasoning"],
            "reasoning": value["reasoning"],
        }
        if keep_as_new:
            proposal[new_entity_type] = new_entity_proposal
        else:
            proposal[rejected_new_entity_type] = new_entity_proposal
            proposal["needs_retry"] = True
        proposal[proposed_existing_entity_type] = None

    return proposal


def apply_review_answer(proposal: FileProposal, answer: dict[str, Any]) -> FileProposal:
    """Act according to the review answer for a proposal. Either keep proposed value or mark as rejected for later use"""
    if answer["accept_all"]:
        proposal["confirmed"] = True
        proposal["needs_retry"] = False
        return proposal

    needs_retry = False

    kept_tags: list = []
    rejected_tags = list(proposal.get("rejected_existing_tags") or [])

    proposed_userrejected_match = zip(proposal["proposed_existing_tags"], answer["proposed_existing_tags"])

    for tag, accepted in proposed_userrejected_match:
        if accepted:
            kept_tags.append(tag)
        else:
            rejected_tags.append(tag)

        needs_retry |= not accepted
    proposal["proposed_existing_tags"] = kept_tags
    proposal["rejected_existing_tags"] = rejected_tags

    kept_new_tags: list = []
    rejected_new_tags = list(proposal.get("rejected_new_tags") or [])
    for tag, accepted in zip(proposal.get("proposed_new_tags") or [], answer["proposed_new_tags"]):
        if accepted:
            kept_new_tags.append(tag)
        else:
            rejected_new_tags.append(tag)

        needs_retry |= not accepted
    proposal["proposed_new_tags"] = kept_new_tags or None
    proposal["rejected_new_tags"] = rejected_new_tags

    for proposal_entity_type, rejected_entity_type in (
        ("proposed_existing_correspondent", "rejected_existing_correspondent"),
        ("proposed_new_correspondent", "rejected_new_correspondent"),
        ("proposed_existing_document_type", "rejected_existing_document_type"),
        ("proposed_new_document_type", "rejected_new_document_type"),
    ):
        value = proposal.get(proposal_entity_type)
        accepted = answer.get(proposal_entity_type)
        if value is not None and not accepted:
            proposal[rejected_entity_type] = value
            proposal[proposal_entity_type] = None
            needs_retry = True

    proposal["needs_retry"] = needs_retry
    proposal["confirmed"] = not needs_retry
    return proposal


def collect_hallucination_answer(hallucinated_proposal: dict[str, Any]) -> bool:
    """Turns a 'hallucination' interrupt into a y/n question for the user."""
    entity = hallucinated_proposal["entity"]
    entity_type = hallucinated_proposal["entity_type"]
    print(f'\nProposal for "{hallucinated_proposal["filename"]}" used a {entity_type} not found in Paperless-ngx (likely a hallucination): "{entity["name"]}".')
    return ask_yes_no("Treat it as a new entity instead of rejecting it?")


def collect_review_answer(proposal_to_verify: dict[str, Any]) -> dict[str, Any]:
    """Turns a 'verify' interrupt into the full accept/reject Q&A for one file's proposal"""
    filename = proposal_to_verify["filename"]
    proposal = proposal_to_verify["proposal"]
    print_proposal((filename, proposal))

    if ask_yes_no("\nDo you fully accept the proposed assignments and new entities?"):
        return {"accept_all": True}

    answer: dict[str, Any] = {"accept_all": False}

    answer["proposed_existing_tags"] = [
        ask_yes_no(f'  Keep tag "{tag["name"]}" (confidence={tag["confidence"]:.2f})?') for tag in proposal["proposed_existing_tags"]
    ]
    answer["proposed_new_tags"] = [
        ask_yes_no(f'  Add new tag "{tag["name"]}" - {tag["description"]}?') for tag in proposal.get("proposed_new_tags") or []
    ]

    for proposal_key, label in (
        ("proposed_existing_correspondent", "correspondent"),
        ("proposed_new_correspondent", "new correspondent"),
        ("proposed_existing_document_type", "document type"),
        ("proposed_new_document_type", "new document type"),
    ):
        value = proposal.get(proposal_key)
        answer[proposal_key] = ask_yes_no(f'  Accept {label} "{value["name"]}"?') if value else None

    return answer
