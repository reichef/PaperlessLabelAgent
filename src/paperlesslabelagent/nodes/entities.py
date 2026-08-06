import os
from typing import Any

from paperlesslabelagent.state import AgentState
from paperlesslabelagent.tools.paperlesstools import build_summary_text, list_tags_correspondents_and_document_types
from paperlesslabelagent.tools.pdftools import extractTextFromPDF


def fetch_existing_entities(state: AgentState) -> dict[str, Any]:
    """Step 1: Read tags, correspondents and document types from Paperless-ngx."""

    existing_entities = list_tags_correspondents_and_document_types(os.getenv("API_URL"), os.getenv("ACCOUNT"), os.getenv("PASSWORD"))

    print(f"Existing entities fetched from Paperless-ngx:\n{build_summary_text(existing_entities)}")

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
