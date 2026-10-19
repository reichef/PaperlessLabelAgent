# Iterative Eager Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upload each confirmed document to Paperless-ngx immediately in the iterative strategy and move it out of the input folder, so an aborted run neither loses nor redoes confirmed work.

**Architecture:** `persist_file_proposals` (shared `core` node) moves each uploaded file to `<INPUT_FOLDER>/processed/` unless `DELETE_INPUT_FILES_AFTER_UPLOAD=true`, which makes the input folder the record of remaining work. `load_documents` ignores subfolders so `processed/` is never picked up. The iterative graph runs `persist_new_entities` → `persist_file_proposals` after every `review_current_proposal`, and only then routes to the next document or `END`.

**Tech Stack:** Python 3.12+, LangGraph (`InMemorySaver` stays), pytest 9.1.1 (new dev dependency).

**Spec:** `docs/specs/2026-10-17-iterative-eager-persistence-design.md`

## Global Constraints

- The processed folder is fixed to `<INPUT_FOLDER>/processed` — no new config variable.
- `DELETE_INPUT_FILES_AFTER_UPLOAD=true` → delete (as today); otherwise → move to `processed/`.
- Name clash in `processed/`: append ` (n)` before the extension (`invoice.pdf` → `invoice (1).pdf`, `invoice (2).pdf`, …). Never overwrite an existing file there.
- Failed entity creation or upload raises and stops the run — no retry logic. The file stays in the input folder.
- No durable checkpointer, no `run_interactive` changes. `strategies/sequential/graph.py` is not rewired (it inherits the moving behavior through `core`).
- `pyproject.toml` gets exactly `[project.optional-dependencies] dev = ["pytest>=9.1.1"]`, identical to branch `perf/classification-prefix-caching`.
- Behavior-specific names for functions, parameters and files (AGENTS.md) — no generic names like `payload`, `helper`, `data`.
- Python interpreter for every command: `.venv/Scripts/python.exe`, run from the repo root `/mnt/c/Users/Frederik/Documents/Arbeitsplatz/git/own/PaperlessLabelAgent` (a Windows venv reachable from WSL; pytest 9.1.1 is already installed and the package is editable-installed).
- **Never touch the real `INPUT_FOLDER`.** The local `.env` sets `DELETE_INPUT_FILES_AFTER_UPLOAD=true` and real credentials, and `core/config.py` calls `load_dotenv(override=True)` at import time, which copies them into `os.environ`. Every test uses pytest's `tmp_path`, and the autouse fixture from Task 1 forces mock mode and unsets the delete flag.
- The mock entity files in `test/paperless-instance-mock/` are empty (0 bytes) in git, so `load_mock_entities()` cannot be used. Tests that need existing entities monkeypatch `list_tags_correspondents_and_document_types` instead.
- Commit messages end with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Repeated name clashes (`processed/` already has `invoice.pdf` and `invoice (1).pdf`) → the new file becomes `invoice (2).pdf`, earlier files untouched. Pinned in Task 1.
2. Filenames with several dots (`scan.2024.pdf`) → suffix goes before the last extension only: `scan.2024 (1).pdf`. Pinned in Task 1.
3. Upload raises (Paperless-ngx down / HTTP error) → exception propagates and the file stays in the input folder, not moved or deleted. Pinned in Task 1.
4. Delete flag set → the file is gone and no `processed/` folder is created. Pinned in Task 1.
5. Restart after abort with `processed/` already present in the input folder → it is neither classified nor crashes the extractor, and only the remaining files are processed. Pinned in Task 2 (unit) and Task 3 (end-to-end).

---

### Task 1: Move uploaded files to `processed/` in `persist_file_proposals`

**Files:**
- Modify: `pyproject.toml` (add dev dependency)
- Create: `test/paperlesslabelagent/conftest.py`
- Create: `test/paperlesslabelagent/core/nodes/test_resultpersistence.py`
- Modify: `src/paperlesslabelagent/core/nodes/resultpersistence.py` (imports, new `_move_to_processed_folder`, end of `persist_file_proposals`, its precondition comment)
- Modify: `README.md` (row for `DELETE_INPUT_FILES_AFTER_UPLOAD`)

**Interfaces:**
- Consumes: `persist_file_proposals(state: AgentState) -> dict[str, Any]` (existing; returns `{"upload_results": {...}}`), `ExistingMatch` from `paperlesslabelagent.core.state`.
- Produces:
  - `PROCESSED_FOLDER_NAME = "processed"` in `paperlesslabelagent.core.nodes.resultpersistence`.
  - `_move_to_processed_folder(file_path: str, input_folder: str) -> str` (returns the target path).
  - autouse fixture `mock_paperless_environment` in `test/paperlesslabelagent/conftest.py` (applies to every test under `test/paperlesslabelagent/`).

- [ ] **Step 1: Add the pytest dev dependency**

In `pyproject.toml`, insert between the `dependencies = [...]` block and `[tool.setuptools.packages.find]`:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=9.1.1",
]

```

- [ ] **Step 2: Add the shared test environment fixture**

Create `test/paperlesslabelagent/conftest.py`:

```python
import pytest


@pytest.fixture(autouse=True)
def mock_paperless_environment(monkeypatch):
    """core/config.py runs load_dotenv(override=True) at import time, which copies the developer's
    real .env (real credentials, DELETE_INPUT_FILES_AFTER_UPLOAD=true) into os.environ. Force mock
    mode and the documented delete default for every test, so no test can reach a real
    Paperless-ngx instance or delete files by accident. Tests that need other values override
    them with their own monkeypatch.setenv calls."""
    monkeypatch.setenv("ACCOUNT", "mock")
    monkeypatch.setenv("PASSWORD", "mock")
    monkeypatch.setenv("API_URL", "")
    monkeypatch.delenv("DELETE_INPUT_FILES_AFTER_UPLOAD", raising=False)
```

- [ ] **Step 3: Write the tests**

Create `test/paperlesslabelagent/core/nodes/test_resultpersistence.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify the new behavior is missing**

Run: `.venv/Scripts/python.exe -m pytest test/paperlesslabelagent/core/nodes/test_resultpersistence.py -v`

Expected: `test_confirmed_file_is_moved_to_processed_folder_when_delete_flag_unset`, `test_name_clash_in_processed_folder_gets_next_free_numeric_suffix` and `test_name_clash_suffix_goes_before_last_extension_for_dotted_filename` FAIL (file still in the input folder / target missing). The other three PASS already — they pin today's delete, skip-unconfirmed and raise-on-failure behavior so it isn't broken.

- [ ] **Step 5: Implement moving to the processed folder**

In `src/paperlesslabelagent/core/nodes/resultpersistence.py`:

Add `import shutil` below `import os`:

```python
import os
import shutil
from dataclasses import replace
```

Below the `ENTITY_API_NAMES = {...}` block, add:

```python
PROCESSED_FOLDER_NAME = "processed"
```

Below `_simulate_upload`, add:

```python
def _move_to_processed_folder(file_path: str, input_folder: str) -> str:
    """Moves an uploaded file into <input_folder>/processed, so the input folder only ever holds
    the work that remains and a restarted run doesn't pick the file up again. If a file of the
    same name is already there, a numeric suffix is appended before the extension
    ("invoice (1).pdf") instead of overwriting the earlier original. Returns the target path."""
    processed_folder = os.path.join(input_folder, PROCESSED_FOLDER_NAME)
    os.makedirs(processed_folder, exist_ok=True)

    filename = os.path.basename(file_path)
    stem, extension = os.path.splitext(filename)
    target_path = os.path.join(processed_folder, filename)
    clash_suffix = 1
    while os.path.exists(target_path):
        target_path = os.path.join(processed_folder, f"{stem} ({clash_suffix}){extension}")
        clash_suffix += 1

    shutil.move(file_path, target_path)
    return target_path
```

At the end of the loop in `persist_file_proposals`, replace:

```python
        if delete_after_upload:
            os.remove(file_path)
```

with:

```python
        if delete_after_upload:
            os.remove(file_path)
        else:
            _move_to_processed_folder(file_path, state["input_folder"])
```

In the comment block above `persist_file_proposals`, replace the line

```python
# - Only files are uploaded only if their proposal is confirmed.
```

with:

```python
# - Only files are uploaded only if their proposal is confirmed.
# - Each uploaded file is removed from the input folder: deleted if DELETE_INPUT_FILES_AFTER_UPLOAD=true, otherwise moved to <input_folder>/processed.
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest test/paperlesslabelagent/core/nodes/test_resultpersistence.py -v`

Expected: 6 passed.

- [ ] **Step 7: Document the behavior in the README**

In `README.md`, replace the row

```markdown
| `DELETE_INPUT_FILES_AFTER_UPLOAD` | Set to `true` to remove a file from `INPUT_FOLDER` once it has been uploaded to Paperless-ngx (default: `false`, i.e. files are kept) |
```

with:

```markdown
| `DELETE_INPUT_FILES_AFTER_UPLOAD` | Set to `true` to delete a file from `INPUT_FOLDER` once it has been uploaded to Paperless-ngx. Otherwise (default: `false`) the file is moved to `INPUT_FOLDER/processed/`; if a file of that name is already there, a numeric suffix is added (e.g. `invoice (1).pdf`) |
```

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml test/paperlesslabelagent/conftest.py test/paperlesslabelagent/core/nodes/test_resultpersistence.py src/paperlesslabelagent/core/nodes/resultpersistence.py README.md
git commit -m "Move uploaded files to INPUT_FOLDER/processed unless deletion is enabled

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `load_documents` skips subfolders

**Files:**
- Create: `test/paperlesslabelagent/core/nodes/test_entities.py`
- Modify: `src/paperlesslabelagent/core/nodes/entities.py` (`load_documents`, loop at lines 27-31)

**Interfaces:**
- Consumes: `load_documents(state: AgentState) -> dict[str, Any]` (existing; returns `{"file_texts": {filename: text}}`), autouse fixture `mock_paperless_environment` (Task 1).
- Produces: `load_documents` returns only regular files of the input folder; directories such as `processed/` are never passed to `extractTextFromPDF`.

- [ ] **Step 1: Write the failing test**

Create `test/paperlesslabelagent/core/nodes/test_entities.py`:

```python
from paperlesslabelagent.core.nodes import entities as entities_module
from paperlesslabelagent.core.nodes.entities import load_documents


def test_load_documents_skips_processed_subfolder(tmp_path, monkeypatch):
    (tmp_path / "invoice.pdf").write_bytes(b"%PDF-1.4 invoice")
    (tmp_path / "processed").mkdir()
    (tmp_path / "processed" / "earlier.pdf").write_bytes(b"%PDF-1.4 earlier")

    extracted_filenames = []

    def fake_extract_text_from_pdf(filename, file_path):
        extracted_filenames.append(filename)
        return f"text of {filename}"

    monkeypatch.setattr(entities_module, "extractTextFromPDF", fake_extract_text_from_pdf)

    result = load_documents({"input_folder": str(tmp_path)})

    assert result == {"file_texts": {"invoice.pdf": "text of invoice.pdf"}}
    assert extracted_filenames == ["invoice.pdf"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest test/paperlesslabelagent/core/nodes/test_entities.py -v`

Expected: FAIL — `file_texts` also contains `"processed"` and `extracted_filenames` contains `"processed"`.

- [ ] **Step 3: Skip non-files in `load_documents`**

In `src/paperlesslabelagent/core/nodes/entities.py`, replace the loop in `load_documents`:

```python
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # TODO We can add other file types supported by Paperless-ngx here, but for now, we will just handle PDF files.
        file_texts[filename] = extractTextFromPDF(filename, file_path)
```

with:

```python
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if not os.path.isfile(file_path):
            continue  # e.g. the processed/ folder persist_file_proposals moves uploaded files into

        # TODO We can add other file types supported by Paperless-ngx here, but for now, we will just handle PDF files.
        file_texts[filename] = extractTextFromPDF(filename, file_path)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest test/paperlesslabelagent -v`

Expected: 7 passed (6 from Task 1, 1 new).

- [ ] **Step 5: Commit**

```bash
git add test/paperlesslabelagent/core/nodes/test_entities.py src/paperlesslabelagent/core/nodes/entities.py
git commit -m "Skip subfolders such as processed/ when loading input documents

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Persist each document right after its review in the iterative graph

**Files:**
- Create: `test/paperlesslabelagent/strategies/iterative/test_graph.py`
- Modify: `src/paperlesslabelagent/strategies/iterative/graph.py` (`route_after_review` and the edge definitions)
- Modify: `README.md` (Usage section)

**Interfaces:**
- Consumes:
  - `persist_file_proposals` moving files to `processed/` (Task 1), `load_documents` skipping subfolders (Task 2), autouse fixture `mock_paperless_environment` (Task 1).
  - `graph` (compiled, module-level, `InMemorySaver`) from `paperlesslabelagent.strategies.iterative.graph`.
  - Interrupt contract: `result["__interrupt__"][0].value == {"kind": "verify", "filename": str, "proposal": ...}`; resume with `Command(resume={"accept_all": True})` to confirm.
  - Monkeypatch targets: `paperlesslabelagent.strategies.iterative.nodes.classify_document(filename, text, existing_entities, **rejected_entities)`, `paperlesslabelagent.core.nodes.entities.list_tags_correspondents_and_document_types(api_url, account, password)`, `paperlesslabelagent.core.nodes.entities.extractTextFromPDF(filename, file_path)`.
- Produces: `route_after_persistence(state: IterativeAgentState) -> str` (replaces `route_after_review`), returning `"classify_current_document"` or `END`.

- [ ] **Step 1: Write the failing end-to-end test**

Create `test/paperlesslabelagent/strategies/iterative/test_graph.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest test/paperlesslabelagent/strategies/iterative/test_graph.py -v`

Expected: FAIL at `assert (processed_folder / confirmed_filename).is_file()` — today the graph only persists after the whole loop, so nothing is uploaded or moved while the second document awaits review.

- [ ] **Step 3: Rewire the iterative graph**

In `src/paperlesslabelagent/strategies/iterative/graph.py`, replace `route_after_review`:

```python
def route_after_review(state: IterativeAgentState) -> str:
    """Iteratively performs document classification and user-review per document.
    Uses accepted entities (tags, document_types and correspondents) in classification of next documents to avoid different syntax for same semantics.
    This avoids a merging step for entities before persistence.
    """
    return "persist_new_entities" if select_next_document(state) is None else "classify_current_document"
```

with:

```python
def route_after_persistence(state: IterativeAgentState) -> str:
    """Iteratively performs document classification, user-review and persistence per document.
    Uses accepted entities (tags, document_types and correspondents) in classification of next documents to avoid different syntax for same semantics.
    This avoids a merging step for entities before persistence.
    Each confirmed document is persisted (and moved out of the input folder) before the next one
    is classified, so aborting the run neither loses confirmed work nor makes a restart redo it.
    """
    return END if select_next_document(state) is None else "classify_current_document"
```

Replace the edge definitions:

```python
workflow.add_edge("classify_current_document", "review_current_proposal")
workflow.add_conditional_edges("review_current_proposal", route_after_review,
    {"classify_current_document": "classify_current_document", "persist_new_entities": "persist_new_entities"},
)
workflow.add_edge("persist_new_entities", "persist_file_proposals")
workflow.add_edge("persist_file_proposals", END)
```

with:

```python
workflow.add_edge("classify_current_document", "review_current_proposal")
# Persist after every review, not only after the last document. On a retry (not confirmed) both
# persist nodes are no-ops: persist_file_proposals skips unconfirmed proposals and
# persist_new_entities only finds placeholder ids that confirmation added.
workflow.add_edge("review_current_proposal", "persist_new_entities")
workflow.add_edge("persist_new_entities", "persist_file_proposals")
workflow.add_conditional_edges("persist_file_proposals", route_after_persistence,
    {"classify_current_document": "classify_current_document", END: END},
)
```

- [ ] **Step 4: Run the full test suite**

Run: `.venv/Scripts/python.exe -m pytest test/paperlesslabelagent -v`

Expected: 8 passed.

- [ ] **Step 5: Document eager persistence in the README**

In `README.md`, directly after the paragraph that starts with `The agent will fetch your existing entities, process every PDF in `INPUT_FOLDER`` (in the Usage section), add:

```markdown

With the iterative strategy, each document is uploaded to Paperless-ngx as soon as you confirm it, and then deleted or moved to `INPUT_FOLDER/processed/` (see `DELETE_INPUT_FILES_AFTER_UPLOAD`). If the agent is aborted, everything confirmed so far is already in Paperless-ngx and out of `INPUT_FOLDER`; start the agent again to continue with the remaining files. A confirmed document cannot be undone from within the agent — correct it directly in Paperless-ngx.
```

- [ ] **Step 6: Commit**

```bash
git add test/paperlesslabelagent/strategies/iterative/test_graph.py src/paperlesslabelagent/strategies/iterative/graph.py README.md
git commit -m "Persist each confirmed document immediately in the iterative strategy

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
