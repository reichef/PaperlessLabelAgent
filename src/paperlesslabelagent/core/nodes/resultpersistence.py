import os
from typing import Any

from paperlesslabelagent.core.state import AgentState
from paperlesslabelagent.core.nodes.paperlesstools import (
    TAGS_API_NAME, CORRESPONDENTS_API_NAME, DOCUMENT_TYPES_API_NAME,
    create_tag, create_correspondent, create_document_type, uses_mock_entities,
)

ENTITY_API_NAMES = {
    TAGS_API_NAME: create_tag,
    CORRESPONDENTS_API_NAME: create_correspondent,
    DOCUMENT_TYPES_API_NAME: create_document_type,
}


def _simulate_create(category_key: str, name: str, existing_entities: dict[str, Any]) -> dict[str, Any]:
    """Mock-mode stand-in for create_tag/create_correspondent/create_document_type: no
    API_URL to call, so assign the current max real id in this category + 1 instead."""
    results = existing_entities.get(category_key, {}).get("results", [])
    real_ids = [item["id"] for item in results if item["id"] > 0]
    return {"id": max(real_ids, default=0) + 1, "name": name}


def persist_new_entities(state: AgentState) -> dict[str, Any]:
    """Creates every confirmed new entity (negative placeholder id in existingEntities) in
    Paperless-ngx for real, replacing its placeholder id with the real, server-assigned one.
    """
    ACCOUNT, PASSWORD, API_URL = os.getenv("ACCOUNT"), os.getenv("PASSWORD"), os.getenv("API_URL")
    is_mock = uses_mock_entities(ACCOUNT, PASSWORD)

    existing_entities = {key: {**value, "results": list(value.get("results", []))} for key, value in state["existingEntities"].items()}

    for category_key, create in ENTITY_API_NAMES.items():
        for entry in existing_entities.get(category_key, {}).get("results", []):
            if entry["id"] >= 0:
                continue  # already a real Paperless-ngx entity
            created = _simulate_create(category_key, entry["name"], existing_entities) if is_mock \
                else create(entry["name"], API_URL, ACCOUNT, PASSWORD)
            entry["id"] = created["id"]

    return {"existingEntities": existing_entities}
