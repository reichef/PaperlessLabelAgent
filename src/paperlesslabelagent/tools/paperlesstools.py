import json
from pathlib import Path
from typing import Any

import requests

MOCK_DIR = Path(__file__).resolve().parents[3] / "test"


def uses_mock_entities(ACCOUNT: str, PASSWORD: str) -> bool:
    return "mock" in (ACCOUNT or "") and "mock" in (PASSWORD or "")


def load_mock_entities() -> dict[str, Any]:
    return {
        "labels": json.loads((MOCK_DIR / "paperless-instance-mock" / "paperless_entity_mock_tags").read_text(encoding="utf-8")),
        "correspondents": json.loads((MOCK_DIR / "paperless-instance-mock" / "paperless_entity_mock_correspondents").read_text(encoding="utf-8")),
        "document_types": json.loads((MOCK_DIR / "paperless-instance-mock" / "paperless_entity_mock_documenttypes").read_text(encoding="utf-8")),
    }


def list_tags_correspondents_and_document_types(API_URL: str, ACCOUNT: str, PASSWORD: str) -> dict[str, Any]:
    """Returns labels, correspondents and document types from Paperless-ngx.

    If ACCOUNT and PASSWORD both contain the string "mock", the content of the mock files under test/ are
    returned instead, so the agent can be tested without a Paperless-ngx instance.
    """
    if uses_mock_entities(ACCOUNT, PASSWORD):
        return load_mock_entities()

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

def create_tag(tag: str, API_URL: str, ACCOUNT: str, PASSWORD: str) -> dict[str, Any]:
    """Creates a new tag in Paperless-ngx and returns the created tag record (including its id)."""
    if not API_URL:
        raise RuntimeError("API_URL is not set. Please check the .env file.")

    tag_url = f"{API_URL}/tags/"
    sendTag = {"name": tag}

    response = requests.post(tag_url, json=sendTag, auth=(ACCOUNT, PASSWORD), timeout=10)
    response.raise_for_status()

    return response.json()

def create_correspondent(correspondent: str, API_URL: str, ACCOUNT: str, PASSWORD: str) -> dict[str, Any]:
    """Creates a new correspondent in Paperless-ngx and returns the created record (including its id)."""
    if not API_URL:
        raise RuntimeError("API_URL is not set. Please check the .env file.")

    correspondent_url = f"{API_URL}/correspondents/"
    sendCorrespondent = {"name": correspondent}

    response = requests.post(correspondent_url, json=sendCorrespondent, auth=(ACCOUNT, PASSWORD), timeout=10)
    response.raise_for_status()

    return response.json()

def create_document_type(document_type: str, API_URL: str, ACCOUNT: str, PASSWORD: str) -> dict[str, Any]:
    """Creates a new document type in Paperless-ngx and returns the created record (including its id)."""
    if not API_URL:
        raise RuntimeError("API_URL is not set. Please check the .env file.")

    document_type_url = f"{API_URL}/document_types/"
    sendDocumentType = {"name": document_type}

    response = requests.post(document_type_url, json=sendDocumentType, auth=(ACCOUNT, PASSWORD), timeout=10)
    response.raise_for_status()

    return response.json()

def upload_document(file_path: str, API_URL: str, ACCOUNT: str, PASSWORD: str, tag_ids: list[int], correspondent_id: int, document_type_id: int,) -> dict[str, Any]:
    """Uploads a document to Paperless-ngx with the given tags,correspondent and document type attached."""
    if not API_URL:
        raise RuntimeError("API_URL is not set. Please check the .env file.")

    upload_url = f"{API_URL}/documents/post_document/"

    data: list[tuple[str, str]] = [("tags", str(tag_id)) for tag_id in tag_ids or []]
    if correspondent_id is not None:
        data.append(("correspondent", str(correspondent_id)))
    if document_type_id is not None:
        data.append(("document_type", str(document_type_id)))

    with open(file_path, "rb") as file:
        files = {"document": file}
        response = requests.post(
            upload_url, files=files, data=data, auth=(ACCOUNT, PASSWORD), timeout=30
        )
        response.raise_for_status()

    return {"task_id": response.json()}