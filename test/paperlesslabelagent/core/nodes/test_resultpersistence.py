import pytest
import requests

from paperlesslabelagent.core.nodes import resultpersistence as resultpersistence_module
from paperlesslabelagent.core.nodes.resultpersistence import persist_file_proposals
from paperlesslabelagent.core.state import ExistingMatch


def _file_proposal(filename: str, confirmed: bool) -> dict:
    return {
        "filename": filename,
        "proposed_existing_tags": [ExistingMatch(name="Invoice", confidence=0.9, reasoning="mentions an amount due", id=1)],
        "proposed_existing_correspondent": None,
        "proposed_existing_document_type": None,
        "proposed_new_tags": None,
        "proposed_new_correspondent": None,
        "proposed_new_document_type": None,
        "confirmed": confirmed,
        "needs_retry": False,
        "rejected_existing_tags": [],
        "rejected_existing_correspondent": None,
        "rejected_existing_document_type": None,
        "rejected_new_tags": None,
        "rejected_new_correspondent": None,
        "rejected_new_document_type": None,
    }


def _persistence_state(input_folder, proposals: dict) -> dict:
    return {"input_folder": str(input_folder), "proposals": proposals, "upload_results": {}}


def test_confirmed_file_is_moved_to_processed_folder_when_delete_flag_unset(tmp_path):
    (tmp_path / "invoice.pdf").write_bytes(b"%PDF-1.4 invoice")

    result = persist_file_proposals(_persistence_state(tmp_path, {"invoice.pdf": _file_proposal("invoice.pdf", confirmed=True)}))

    assert result["upload_results"] == {"invoice.pdf": {"task_id": "mock-task-invoice.pdf"}}
    assert not (tmp_path / "invoice.pdf").exists()
    assert (tmp_path / "processed" / "invoice.pdf").read_bytes() == b"%PDF-1.4 invoice"


def test_confirmed_file_is_deleted_without_processed_folder_when_delete_flag_set(tmp_path, monkeypatch):
    monkeypatch.setenv("DELETE_INPUT_FILES_AFTER_UPLOAD", "true")
    (tmp_path / "invoice.pdf").write_bytes(b"%PDF-1.4 invoice")

    persist_file_proposals(_persistence_state(tmp_path, {"invoice.pdf": _file_proposal("invoice.pdf", confirmed=True)}))

    assert not (tmp_path / "invoice.pdf").exists()
    assert not (tmp_path / "processed").exists()


def test_name_clash_in_processed_folder_gets_next_free_numeric_suffix(tmp_path):
    processed_folder = tmp_path / "processed"
    processed_folder.mkdir()
    (processed_folder / "invoice.pdf").write_bytes(b"earliest")
    (processed_folder / "invoice (1).pdf").write_bytes(b"earlier")
    (tmp_path / "invoice.pdf").write_bytes(b"%PDF-1.4 newest")

    persist_file_proposals(_persistence_state(tmp_path, {"invoice.pdf": _file_proposal("invoice.pdf", confirmed=True)}))

    assert (processed_folder / "invoice.pdf").read_bytes() == b"earliest"
    assert (processed_folder / "invoice (1).pdf").read_bytes() == b"earlier"
    assert (processed_folder / "invoice (2).pdf").read_bytes() == b"%PDF-1.4 newest"


def test_name_clash_suffix_goes_before_last_extension_for_dotted_filename(tmp_path):
    processed_folder = tmp_path / "processed"
    processed_folder.mkdir()
    (processed_folder / "scan.2024.pdf").write_bytes(b"earlier")
    (tmp_path / "scan.2024.pdf").write_bytes(b"%PDF-1.4 newer")

    persist_file_proposals(_persistence_state(tmp_path, {"scan.2024.pdf": _file_proposal("scan.2024.pdf", confirmed=True)}))

    assert (processed_folder / "scan.2024 (1).pdf").read_bytes() == b"%PDF-1.4 newer"


def test_unconfirmed_file_stays_in_input_folder(tmp_path):
    (tmp_path / "invoice.pdf").write_bytes(b"%PDF-1.4 invoice")

    result = persist_file_proposals(_persistence_state(tmp_path, {"invoice.pdf": _file_proposal("invoice.pdf", confirmed=False)}))

    assert result["upload_results"] == {}
    assert (tmp_path / "invoice.pdf").exists()
    assert not (tmp_path / "processed").exists()


def test_failed_upload_raises_and_leaves_file_in_input_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("ACCOUNT", "paperless-user")
    monkeypatch.setenv("PASSWORD", "paperless-password")
    monkeypatch.setenv("API_URL", "http://paperless.invalid/api")

    def failing_upload_document(file_path, api_url, account, password, tag_ids, correspondent_id, document_type_id):
        raise requests.HTTPError("503 Service Unavailable")

    monkeypatch.setattr(resultpersistence_module, "upload_document", failing_upload_document)
    (tmp_path / "invoice.pdf").write_bytes(b"%PDF-1.4 invoice")

    with pytest.raises(requests.HTTPError):
        persist_file_proposals(_persistence_state(tmp_path, {"invoice.pdf": _file_proposal("invoice.pdf", confirmed=True)}))

    assert (tmp_path / "invoice.pdf").exists()
    assert not (tmp_path / "processed").exists()
