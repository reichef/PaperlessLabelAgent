import os

import pymupdf
from langchain_ollama import ChatOllama

from paperlesslabelagent.config import MODEL, OCR_LANGUAGES, TESSDATA_PATH

summarizer = ChatOllama(model=MODEL, num_ctx=32768, temperature=0.3)


SUMMARIZER_SYSTEM_PROMPT = """You are an assistant that provides the parts of the given document text to support the extraction of labels, document_types and correspondents for Paperless-ngx.

Rules: 
- Provide the text of the document, focusing on information that helps identify the appropriate labels, document_types and correspondents. Avoid including any irrelevant details or personal opinions. 
- Only provide the text that is relevant for classification
- Do not addy any additional commentary or explanation.
- Do not yourself provide the labels, document_types or correspondents. Only provide the text that is relevant for classification."""

def summarize_document_text(text: str) -> str:
    """
    Summarizes the already-extracted text of a document, keeping only the information relevant
    to choosing tags, a correspondent and a document type for Paperless-ngx. Use this when the
    full document text is too large to reason about directly.

    Args:
        text (str): The full extracted text of the document.

    Returns:
        str: A concise summary of the document content.
    """
    result = summarizer.invoke(
        [
            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
    )
    return result.content


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
