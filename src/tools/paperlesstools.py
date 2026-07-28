from typing import Any

import requests
from langchain.tools import tool

@tool
def list_tags_correspondents_and_document_types(API_URL: str, ACCOUNT: str, PASSWORD: str) -> dict[str, Any]:
    """Returns labels, correspondents and document types from Paperless-ngx."""
    if not API_URL:
        raise RuntimeError("API_URL is not set. Please check the .env file.")

    label_url = f"{API_URL}/tags/"
    correspondent_url = f"{API_URL}/correspondents/"
    document_type_url = f"{API_URL}/document_types/"

    response_labels = requests.get(label_url, auth=(ACCOUNT, PASSWORD), timeout=10)
    response_correspondents = requests.get(correspondent_url, auth=(ACCOUNT, PASSWORD), timeout=10)
    response_document_types = requests.get(document_type_url, auth=(ACCOUNT, PASSWORD), timeout=10)

    response_labels.raise_for_status()
    response_correspondents.raise_for_status()
    response_document_types.raise_for_status()

    return {
        "labels": response_labels.json(),
        "correspondents": response_correspondents.json(),
        "document_types": response_document_types.json(),
    }


def build_summary_text(data: dict[str, Any]) -> str:
    """Create a summary text from the data returned by list_tags_correspondents_and_document_types."""
    labels = data.get("labels", {}).get("results", [])
    correspondents = data.get("correspondents", {}).get("results", [])
    document_types = data.get("document_types", {}).get("results", [])

    parts: list[str] = []
    for item in labels:
        parts.append(f"Label: {item.get('name', '')}")
    for item in correspondents:
        parts.append(f"Correspondent: {item.get('name', '')}")
    for item in document_types:
        parts.append(f"Document Type: {item.get('name', '')}")

    return "\n".join(parts)