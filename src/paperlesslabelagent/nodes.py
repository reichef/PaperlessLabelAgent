import os
from typing import Any

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from pymupdf import Document, pymupdf

from paperlesslabelagent.schemas import FileProposalModel
from paperlesslabelagent.state import AgentState, FileProposal
from paperlesslabelagent.tools.paperlesstools import list_tags_correspondents_and_document_types

load_dotenv()


MODEL = os.getenv("MODEL")
TESSDATA_PATH = os.getenv("TESSDATA_PATH")
OCR_LANGUAGES = "deu+eng+fra"


def fetch_existing_entities(state: AgentState) -> dict[str, Any]:
    """Step 1: Read tags, correspondents and document types from Paperless-ngx."""
    existing_entities = list_tags_correspondents_and_document_types(os.getenv("API_URL"), os.getenv("ACCOUNT"), os.getenv("PASSWORD"))
    return {"existingEntities": existing_entities}


def load_documents(state: AgentState) -> dict[str, Any]:
    """Step 2: Analyze the files in the input folder and extract their text content."""
    input_folder = state["input_folder"]

    if not os.path.exists(input_folder):
        raise RuntimeError(f"Input folder '{input_folder}' does not exist.")

    file_texts: dict[str, str] = {}
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # TODO We can add other file types supported by Paperless-ngx here, but for now, we will just handle PDF files.
        file_texts[filename] = extractTextFromPDF(filename, file_path)

    return {"file_texts": file_texts}


def extractTextFromPDF(filename: str, file_path: str) -> str:
    """Extracts text from a PDF file using PyMuPDF."""
    if not os.path.isfile(file_path):
        raise RuntimeError(f"File '{file_path}' does not exist.")

    if not filename.lower().endswith((".pdf")):
        raise RuntimeError(f"File '{file_path}' is not a PDF file.")

    document: Document = pymupdf.open(file_path)
    text = ""

    for page in document:
        textpage = page.get_textpage_ocr(dpi=300, full=False, language=OCR_LANGUAGES, tessdata=TESSDATA_PATH)
        text += page.get_text(textpage=textpage)

    return text


MATCH_SYSTEM_PROMPT = """You are an assistant that labels documents for Paperless-ngx.
For the given document text, decide the best-fitting tags, correspondent and document type.

Rules:
- Only choose an existing tag/correspondent/document type if you are reasonably confident it fits (confidence >= 0.6). Reference it by its exact id.
- Ensure that the proposed tags, correspondent and document type exist in the provided list of existing entities.
- A document can have zero, one or several tags.
- A document has at most one correspondent and at most one document type.
- If no existing tag fits well, propose new tags via `new_tags` instead of forcing a bad match.
- If no existing correspondent or document type fits well, propose one via `new_correspondent` / `new_document_type` instead of forcing a bad match.
- Do not propose a new tag, correspondent or document type if a suitable existing one already exists.
- If you propose a new tag, correspondent or document type, avoid additions to the name like (or similar)
- There must be no (no match) in tag/correspondent/document type without a proposed new tag, new correspondent or new document type.
"""

matcher = ChatOllama(model=MODEL, num_ctx=32768).with_structured_output(FileProposalModel)


def format_entity(entities: dict[str, Any]) -> str:
    items = entities.get("results", [])
    if not items:
        return "(none yet)"
    return "\n".join(f'- id={item["id"]}, name="{item["name"]}"' for item in items)


def build_user_prompt(filename: str, text: str, existing_entities: dict[str, Any]) -> str:
    tags = format_entity(existing_entities.get("labels", {}))
    correspondents = format_entity(existing_entities.get("correspondents", {}))
    document_types = format_entity(existing_entities.get("document_types", {}))

    return f"""Document: {filename}

Existing tags:
{tags}

Existing correspondents:
{correspondents}

Existing document types:
{document_types}

Document text:
{text}
"""


def propose_for_document(state: AgentState) -> dict[str, Any]:
    """Step 3: For each not-yet-confirmed file, propose matching (or new) tags, correspondent and document type."""
    existing_entities = state["existingEntities"]
    proposals: dict[str, FileProposal] = dict(state.get("proposals", {}))

    for filename, text in state["file_texts"].items():
        if proposals.get(filename, {}).get("confirmed"):
            continue

        user_prompt = build_user_prompt(filename, text, existing_entities)
        result: FileProposalModel = matcher.invoke(
            [
                {"role": "system", "content": MATCH_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )

        proposals[filename] = {
            "filename": filename,
            "tags": [tag.model_dump() for tag in result.tags],
            "correspondent": result.correspondent.model_dump() if result.correspondent else None,
            "document_type": result.document_type.model_dump() if result.document_type else None,
            "new_tags": [proposal.model_dump() for proposal in result.new_tags] or None,
            "new_correspondent": result.new_correspondent.model_dump() if result.new_correspondent else None,
            "new_document_type": result.new_document_type.model_dump() if result.new_document_type else None,
            "confirmed": False,
            "needs_retry": False,
        }

    return {"proposals": proposals}


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
    
        new_document_type = proposal[1]["new_document_type"]
        if new_document_type:
            print(f'  New document type proposed: "{new_document_type["name"]}" - {new_document_type["description"]}')

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


