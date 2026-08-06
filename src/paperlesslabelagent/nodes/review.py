from typing import Any

from paperlesslabelagent.state import AgentState


def print_proposal(proposal):
    print(f"\n --- Proposition of assignments of tags, correspondent and document types === {proposal[0]} === as well as new ones labels, correspondents and document types ---")

    if proposal[1]["tags"]:
        for tag in proposal[1]["tags"]:
            print(f'  Tag: "{tag["name"]}" (id={tag["id"]}, confidence={tag["confidence"]:.2f})')
    else:
        print("  Tags: (no match)")

    correspondent = proposal[1]["correspondent"]
    if correspondent:
        print(f'  Correspondent: "{correspondent["name"]}" (id={correspondent["id"]}, confidence={correspondent["confidence"]:.2f})')
    else:
        print("  Correspondent: (no match)")

    document_type = proposal[1]["document_type"]
    if document_type:
        print(f'  Document type: "{document_type["name"]}" (id={document_type["id"]}, confidence={document_type["confidence"]:.2f})')
    else:
        print("  Document type: (no match)")

    for new_tag in proposal[1]["new_tags"] or []:
        print(f'  New tag proposed: "{new_tag["name"]}" - {new_tag["description"]}')

    new_correspondent = proposal[1]["new_correspondent"]
    if new_correspondent:
        print(f'  New correspondent proposed: "{new_correspondent["name"]}" - {new_correspondent["description"]}')
    else:
        print("  New correspondent: (no match)")

    new_document_type = proposal[1]["new_document_type"]
    if new_document_type:
        print(f'  New document type proposed: "{new_document_type["name"]}" - {new_document_type["description"]}')
    else:
        print("  New document type: (no match)")

def print_proposals(state: AgentState) -> dict[str, Any]:
    """Prints every file's proposal to the console (filename + matches only, no document text)."""
    proposals = state.get("proposals", {})

    if not proposals:
        print("No proposals yet.")
        return {}

    for proposal in proposals.items():
        print_proposal(proposal)

    return {}

def check_and_correct_proposals(state: AgentState) -> dict[str, Any]:
    """Checks every proposal for erroneous assignments of hallucinated existing entries"""

    existing_entities = state["existingEntities"]
    tags = existing_entities.get("labels", {}).get("results", [])
    correspondents = existing_entities.get("correspondents", {}).get("results", [])
    document_types = existing_entities.get("document_types", {}).get("results", [])

    correspondent_names = {item["name"] for item in correspondents}
    document_type_names = {item["name"] for item in document_types}
    correspondent_ids = {item["id"] for item in correspondents}
    document_type_ids = {item["id"] for item in document_types}

    proposals = state.get("proposals", {})

    for proposal in proposals.items():

        mappedTags = proposal[1]["tags"]
        mappedCorrespondent = proposal[1]["correspondent"]
        mappedDocumentType = proposal[1]["document_type"]

        #TODO check everything for Erroneous existing tags as well

        if mappedCorrespondent is not None and (mappedCorrespondent["id"] not in correspondent_ids or mappedCorrespondent["name"] not in correspondent_names) :

            print (f'Proposal for {proposal[0]} erroneous (e.g., LLM hallucinated) due to wrong Correspondent, set for retry \n')
            print (f'Should the proposed correspondent be used as new entity (y) or should we retry (n)?')

            userInput : str
            #TODO wait for user input

            if(userInput == '(y)'):
                proposal[1]["new_correspondent"] = proposal[1]["correspondent"]
                proposal[1]["correspondent"] = None
            else:
                proposal[1]["rejected_existing_correspondent"] = proposal[1]["correspondent"]
                proposal[1]["correspondent"] = None
                proposal[1]["needs_retry"] |= True


        if mappedDocumentType is not None and(mappedDocumentType["id"] not in document_type_ids or mappedDocumentType["name"] not in document_type_names):
            print (f'Proposal for {proposal[0]} erroneous (e.g., LLM hallucinated) due to wrong Document Type, set for retry \n')
            print (f'Should the proposed correspondent be used as new entity (y) or should we retry (n)?')

            userInput : str
            #TODO wait for user input

            if(userInput == '(y)'):
                proposal[1]["new_document_type"] = proposal[1]["document_type"]
                proposal[1]["document_type"] = None
            else:
                proposal[1]["rejected_existing_documentType"] = proposal[1]["document_type"]
                proposal[1]["document_type"] = None
                proposal[1]["needs_retry"] |= True




    return {"proposals": proposals}

def user_verify_proposals(state: AgentState):
    """
    Interacts with the user to determine whether the proposals (assigned entities and new entities)
    provided by the AI fit the users oppinion
    """

    proposals = state.get("proposals", {})

    for proposal in proposals.items():

        if proposal.get[1]["needs_retry"] is True:
            continue # Skip already rejected proposals due to type errors

        print_proposal(proposal=proposal)

        proposalFullyAccepted: bool = False

        print (f'Do you fully accept the proposed assignments and new entities?')

        #TODO get response and translate to
