import json
from pathlib import Path
from typing import Any

import requests

MOCK_DIR = Path(__file__).resolve().parents[4] / "test"
TAGS_API_NAME = "tags"
CORRESPONDENTS_API_NAME = "correspondents"
DOCUMENT_TYPES_API_NAME = "document_types"

def uses_mock_entities(ACCOUNT: str, PASSWORD: str) -> bool:
    return "mock" in (ACCOUNT or "") and "mock" in (PASSWORD or "")


def load_mock_entities() -> dict[str, Any]:
    return {
        "tags": json.loads((MOCK_DIR / "paperless-instance-mock" / "paperless_entity_mock_tags").read_text(encoding="utf-8")),
        "correspondents": json.loads((MOCK_DIR / "paperless-instance-mock" / "paperless_entity_mock_correspondents").read_text(encoding="utf-8")),
        "document_types": json.loads((MOCK_DIR / "paperless-instance-mock" / "paperless_entity_mock_documenttypes").read_text(encoding="utf-8")),
    }

def _get_Entities(Entity_API_Name: str, API_URL: str, ACCOUNT: str, PASSWORD: str) -> Any:
    """Gets the Entities with the Entity_API_Name from Paperless-ngx instance""" 
    entity_url = f"{API_URL}/{Entity_API_Name}/"
    response_entities = requests.get(entity_url, auth=(ACCOUNT, PASSWORD), timeout=10)
    response_entities.raise_for_status()
    return response_entities.json()


def list_tags_correspondents_and_document_types(API_URL: str, ACCOUNT: str, PASSWORD: str) -> dict[str, Any]:
    """Returns tags, correspondents and document types from Paperless-ngx.

    If ACCOUNT and PASSWORD both contain the string "mock", the content of the mock files under test/ are
    returned instead, so the agent can be tested without a Paperless-ngx instance.
    """
    if uses_mock_entities(ACCOUNT, PASSWORD):
        return load_mock_entities()

    if not API_URL:
        raise RuntimeError("API_URL is not set. Please check the .env file.")

    return {
        "tags": _get_Entities(Entity_API_Name=TAGS_API_NAME, API_URL=API_URL, ACCOUNT=ACCOUNT, PASSWORD=PASSWORD),
        "correspondents": _get_Entities(Entity_API_Name=CORRESPONDENTS_API_NAME, API_URL=API_URL, ACCOUNT=ACCOUNT, PASSWORD=PASSWORD),
        "document_types": _get_Entities(Entity_API_Name=DOCUMENT_TYPES_API_NAME, API_URL=API_URL, ACCOUNT=ACCOUNT, PASSWORD=PASSWORD),
    }


def build_summary_text(data: dict[str, Any]) -> str:
    """Create a summary text from the data returned by list_tags_correspondents_and_document_types."""
    tags = data.get("tags", {}).get("results", [])
    correspondents = data.get("correspondents", {}).get("results", [])
    document_types = data.get("document_types", {}).get("results", [])

    parts: list[str] = []
    for item in tags:
        parts.append(f"Tag: {item.get('name', '')}")
    for item in correspondents:
        parts.append(f"Correspondent: {item.get('name', '')}")
    for item in document_types:
        parts.append(f"Document Type: {item.get('name', '')}")

    return "\n".join(parts)


def _create_entity(Entity_API_Name: str, name: str, API_URL: str, ACCOUNT: str, PASSWORD: str) -> dict[str, Any]:
    """Creates an entity with the Entity_API_Name in Paperless-ngx and returns the created record (including its id)."""
    if not API_URL:
        raise RuntimeError("API_URL is not set. Please check the .env file.")

    entity_url = f"{API_URL}/{Entity_API_Name}/"
    response = requests.post(entity_url, json={"name": name}, auth=(ACCOUNT, PASSWORD), timeout=10)
    response.raise_for_status()

    return response.json()

def create_tag(tag: str, API_URL: str, ACCOUNT: str, PASSWORD: str) -> dict[str, Any]:
    """Creates a new tag in Paperless-ngx and returns the created tag record (including its id)."""
    return _create_entity(TAGS_API_NAME, tag, API_URL, ACCOUNT, PASSWORD)

def create_correspondent(correspondent: str, API_URL: str, ACCOUNT: str, PASSWORD: str) -> dict[str, Any]:
    """Creates a new correspondent in Paperless-ngx and returns the created record (including its id)."""
    return _create_entity(CORRESPONDENTS_API_NAME, correspondent, API_URL, ACCOUNT, PASSWORD)

def create_document_type(document_type: str, API_URL: str, ACCOUNT: str, PASSWORD: str) -> dict[str, Any]:
    """Creates a new document type in Paperless-ngx and returns the created record (including its id)."""
    return _create_entity(DOCUMENT_TYPES_API_NAME, document_type, API_URL, ACCOUNT, PASSWORD)

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