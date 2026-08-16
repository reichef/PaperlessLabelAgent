import os
from typing import Any
import pymupdf

from paperlesslabelagent.core.config import OCR_LANGUAGES, TESSDATA_PATH

from paperlesslabelagent.core.state import AgentState, EntityType, FileProposal, NewEntityProposal
from paperlesslabelagent.core.nodes.paperlesstools import build_summary_text, list_tags_correspondents_and_document_types


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

    if not filename.lower().endswith(".pdf"):
        raise RuntimeError(f"File '{file_path}' is not a PDF file.")

    document = pymupdf.open(file_path)
    text = ""

    for page in document:
        textpage = page.get_textpage_ocr(dpi=300, full=False, language=OCR_LANGUAGES, tessdata=TESSDATA_PATH)
        text += page.get_text(textpage=textpage)

    return text


def _add_entity_to_pool(
    existing_entities: dict[str, Any],
    confirmed_new_entities: list[NewEntityProposal],
    entity_type: EntityType,
    entity: NewEntityProposal,
) -> None:
    temp_id = -(len(confirmed_new_entities) + 1)
    category_key = f"{entity_type}s"
    existing_entities.setdefault(category_key, {}).setdefault("results", []).append({"id": temp_id, "name": entity["name"]})
    confirmed_new_entities.append(entity)


def merge_confirmed_new_entities(
    existing_entities: dict[str, Any],
    proposal: FileProposal,
    confirmed_new_entities: list[NewEntityProposal],
) -> tuple[dict[str, Any], list[NewEntityProposal]]:
    """Folds every newly-accepted entity on a confirmed proposal into a fresh copy of
    existing_entities (using temporary placeholder ids)."""
    existing_entities = {key: {"results": list(value.get("results", []))} for key, value in existing_entities.items()}
    confirmed_new_entities = list(confirmed_new_entities)

    for tag in proposal.get("proposed_new_tags") or []:
        _add_entity_to_pool(existing_entities, confirmed_new_entities, "tag", tag)
    if correspondent := proposal.get("proposed_new_correspondent"):
        _add_entity_to_pool(existing_entities, confirmed_new_entities, "correspondent", correspondent)
    if document_type := proposal.get("proposed_new_document_type"):
        _add_entity_to_pool(existing_entities, confirmed_new_entities, "document_type", document_type)

    return existing_entities, confirmed_new_entities
