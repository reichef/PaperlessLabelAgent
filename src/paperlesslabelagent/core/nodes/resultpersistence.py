import os
from typing import Any

from paperlesslabelagent.core.state import AgentState
from paperlesslabelagent.core.nodes.paperlesstools import (
    TAGS_API_NAME, CORRESPONDENTS_API_NAME, DOCUMENT_TYPES_API_NAME,
    create_tag, create_correspondent, create_document_type, uses_mock_entities, upload_document,
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


def _simulate_upload(filename: str) -> dict[str, Any]:
    """Mock-mode stand-in for upload_document: no API_URL to call, so fake a task id instead."""
    return {"task_id": f"mock-task-{filename}"}


# Precondition: The state now contains a list of confirmed assignments of existing entities (with positive ids) and new entities (with negative placeholder ids)
# Postcondition: The entities with negative placeholder ids are created in Paperless-ngx by calling the appropriate API functions and their placeholder ids are replaced with the ids assigned by paperless-ngx. The state now contains only positive ids for all entities.
def persist_new_entities(state: AgentState) -> dict[str, Any]:
    """Creates every confirmed new entity (negative placeholder id in existingEntities) in
    Paperless-ngx for real, replacing its placeholder id with the real, server-assigned one.
    """
    ACCOUNT, PASSWORD, API_URL = os.getenv("ACCOUNT"), os.getenv("PASSWORD"), os.getenv("API_URL")
    is_mock = uses_mock_entities(ACCOUNT, PASSWORD)

    existing_entities = {key: {**value, "results": list(value.get("results", []))} for key, value in state["existingEntities"].items()}
    confirmed_new_entities = [dict(entity) for entity in state.get("confirmed_new_entities", [])]
    real_id_by_placeholder: dict[int, int] = {}

    for category_key, create in ENTITY_API_NAMES.items():
        results = existing_entities.get(category_key, {}).get("results", [])
        # Two different files' proposals can each add the same-named new entity as its own
        # placeholder (merge_confirmed_new_entities does not dedupe by name), so track ids by
        # name as we go and only actually create one Paperless-ngx entity per distinct name.
        # TODO: To solve this problem, in a next iteration, we could use objects instead of plain dicts. 
        real_id_by_name = {entry["name"]: entry["id"] for entry in results if entry["id"] >= 0}

        for entry in results:
            if entry["id"] >= 0:
                continue  # already a real Paperless-ngx entity

            if entry["name"] not in real_id_by_name:
                created: dict[str, Any]
                if is_mock:
                    created = _simulate_create(category_key, entry["name"], existing_entities)
                else:
                    created = create(entry["name"], API_URL, ACCOUNT, PASSWORD)

                real_id_by_name[entry["name"]] = created["id"]

            real_id_by_placeholder[entry["id"]] = real_id_by_name[entry["name"]]
            entry["id"] = real_id_by_name[entry["name"]]

    for entity in confirmed_new_entities:
        if entity.get("id") in real_id_by_placeholder:
            entity["id"] = real_id_by_placeholder[entity["id"]]

    return {"existingEntities": existing_entities, "confirmed_new_entities": confirmed_new_entities}


# Preconditions: 
# - The state now contains only confirmed entities that exist in Paperless-ngx (positive ids) 
# - The state now contains a list of file proposals, each with a filename and the assigned tags, correspondent and document type (all with positive ids)
# Postconditions: 
# - The files are uploaded to Paperless-ngx with the assigned tags, correspondent and document type. The state now contains the upload results for each file.
# - Only files are uploaded only if their proposal is confirmed.
def persist_file_proposals(state: AgentState) -> dict[str, Any]:
    """ Persists the file proposals to Paperless-ngx by uploading the files and assigning the tags, correspondents and document types."""
    ACCOUNT, PASSWORD, API_URL = os.getenv("ACCOUNT"), os.getenv("PASSWORD"), os.getenv("API_URL")
    is_mock = uses_mock_entities(ACCOUNT, PASSWORD)
    delete_after_upload = (os.getenv("DELETE_INPUT_FILES_AFTER_UPLOAD") or "").strip().lower() == "true"

    existing_entities = state["existingEntities"]
    tag_ids_by_name = {item["name"]: item["id"] for item in existing_entities.get(TAGS_API_NAME, {}).get("results", [])}
    correspondent_ids_by_name = {item["name"]: item["id"] for item in existing_entities.get(CORRESPONDENTS_API_NAME, {}).get("results", [])}
    document_type_ids_by_name = {item["name"]: item["id"] for item in existing_entities.get(DOCUMENT_TYPES_API_NAME, {}).get("results", [])}

    upload_results = dict(state.get("upload_results", {}))

    for filename, proposal in state.get("proposals", {}).items():
        if not proposal.get("confirmed") or filename in upload_results:
            continue  # skip unconfirmed proposals (e.g. retry limit hit) and files already uploaded (replay safety)

        tag_names = [tag["name"] for tag in proposal.get("proposed_existing_tags") or []]
        tag_names += [tag["name"] for tag in proposal.get("proposed_new_tags") or []]
        tag_ids = [tag_ids_by_name[name] for name in tag_names]

        correspondent = proposal.get("proposed_existing_correspondent") or proposal.get("proposed_new_correspondent")
        correspondent_id = correspondent_ids_by_name[correspondent["name"]] if correspondent else None

        document_type = proposal.get("proposed_existing_document_type") or proposal.get("proposed_new_document_type")
        document_type_id = document_type_ids_by_name[document_type["name"]] if document_type else None

        file_path = os.path.join(state["input_folder"], filename)
        upload_results[filename] = _simulate_upload(filename) if is_mock \
            else upload_document(file_path, API_URL, ACCOUNT, PASSWORD, tag_ids, correspondent_id, document_type_id)

        if delete_after_upload:
            os.remove(file_path)

    return {"upload_results": upload_results}