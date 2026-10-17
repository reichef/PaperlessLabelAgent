import os

import pytest
from langgraph.types import Command

from paperlesslabelagent.core.nodes import entities as entities_module
from paperlesslabelagent.core.state import ExistingMatch, NewEntityProposal
from paperlesslabelagent.strategies.iterative import nodes as iterative_nodes_module
from paperlesslabelagent.strategies.iterative.graph import graph

INPUT_FILENAMES = ["alpha.pdf", "beta.pdf", "gamma.pdf"]


def _paperless_entities() -> dict:
    return {
        "tags": {"results": [{"id": 1, "name": "Invoice"}]},
        "correspondents": {"results": [{"id": 2, "name": "Acme Corp"}]},
        "document_types": {"results": [{"id": 3, "name": "Letter"}]},
    }


class _ClassificationRecorder:
    """Stands in for classify_document (no LLM): records which files were classified and
    proposes the existing tag "Invoice" plus a new tag "Insurance", so confirming a document
    exercises both persist_new_entities and persist_file_proposals."""

    def __init__(self):
        self.classified_filenames: list[str] = []

    def __call__(self, filename, text, existing_entities, **rejected_entities):
        self.classified_filenames.append(filename)
        return {
            "filename": filename,
            "proposed_existing_tags": [ExistingMatch(name="Invoice", confidence=0.9, reasoning="mentions an amount due")],
            "proposed_existing_correspondent": None,
            "proposed_existing_document_type": None,
            "proposed_new_tags": [NewEntityProposal(entity_type="tag", name="Insurance", description="Insurance documents", reasoning="mentions a policy number")],
            "proposed_new_correspondent": None,
            "proposed_new_document_type": None,
            "confirmed": False,
            "needs_retry": False,
            "rejected_existing_tags": [],
            "rejected_existing_correspondent": None,
            "rejected_existing_document_type": None,
            "rejected_new_tags": [],
            "rejected_new_correspondent": None,
            "rejected_new_document_type": None,
        }


@pytest.fixture
def input_folder(tmp_path):
    for filename in INPUT_FILENAMES:
        (tmp_path / filename).write_bytes(b"%PDF-1.4 " + filename.encode())
    return tmp_path


@pytest.fixture
def classification_recorder(monkeypatch):
    recorder = _ClassificationRecorder()
    monkeypatch.setattr(iterative_nodes_module, "classify_document", recorder)
    monkeypatch.setattr(entities_module, "list_tags_correspondents_and_document_types", lambda api_url, account, password: _paperless_entities())
    monkeypatch.setattr(entities_module, "extractTextFromPDF", lambda filename, file_path: f"text of {filename}")
    return recorder


def _filename_awaiting_review(result) -> str:
    interrupt_content = result["__interrupt__"][0].value
    assert interrupt_content["kind"] == "verify"
    return interrupt_content["filename"]


def test_confirmed_document_is_persisted_before_next_review_and_skipped_after_restart(input_folder, classification_recorder):
    processed_folder = input_folder / "processed"

    # First run: confirm one document, then abort while the next one awaits review.
    first_run = {"configurable": {"thread_id": "eager-persistence-first-run"}}
    result = graph.invoke({"input_folder": str(input_folder)}, config=first_run)
    confirmed_filename = _filename_awaiting_review(result)
    result = graph.invoke(Command(resume={"accept_all": True}), config=first_run)
    assert _filename_awaiting_review(result) != confirmed_filename

    assert (processed_folder / confirmed_filename).is_file()
    assert not (input_folder / confirmed_filename).exists()
    first_run_values = graph.get_state(first_run).values
    assert confirmed_filename in first_run_values["upload_results"]
    assert all(tag.id > 0 for tag in first_run_values["proposals"][confirmed_filename]["proposed_new_tags"])

    # Restart: the first run is never resumed; a new thread starts with empty state, like a new process.
    classification_recorder.classified_filenames.clear()
    second_run = {"configurable": {"thread_id": "eager-persistence-second-run"}}
    result = graph.invoke({"input_folder": str(input_folder)}, config=second_run)
    while "__interrupt__" in result:
        _filename_awaiting_review(result)
        result = graph.invoke(Command(resume={"accept_all": True}), config=second_run)

    remaining_filenames = sorted(set(INPUT_FILENAMES) - {confirmed_filename})
    assert sorted(classification_recorder.classified_filenames) == remaining_filenames
    assert sorted(result["upload_results"]) == remaining_filenames
    assert sorted(os.listdir(processed_folder)) == sorted(INPUT_FILENAMES)
    assert os.listdir(input_folder) == ["processed"]
