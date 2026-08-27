# Entity ID Objects Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the scattered `name → id` mapping-table lookups for tags/correspondents/document
types with `ExistingMatch`/`NewEntityProposal` dataclasses that carry a resolved `.id`, plus one
explicit, centralized propagation step for new-entity real ids.

**Architecture:** `core/state.py`'s `ExistingMatch` and `NewEntityProposal` change from
`TypedDict` to `@dataclass` with a mutable `id: int | None = None`. Existing-entity ids are
resolved once, as today, in `check_and_correct_single_proposal`. New-entity ids get their
placeholder set by mutating the proposal's own object in place (`core/nodes/entities.py`), and
their final real id propagated explicitly by placeholder-id correlation in
`persist_new_entities` (`core/nodes/resultpersistence.py`) — not via shared object references,
which don't survive LangGraph's per-channel checkpointing under this codebase's
`interrupt()`-heavy flow. Consumers (`persist_file_proposals`) then just read `.id`.

**Tech Stack:** Python 3.12+, `dataclasses` (stdlib), LangGraph, no new dependencies.

**Spec:** `docs/specs/2026-08-26-entity-id-objects-design.md`

## Global Constraints

- The `existingEntities` pool (`state["existingEntities"][category]["results"]`) stays plain
  `{"id": ..., "name": ...}` dicts — it mirrors the Paperless-ngx API response shape. Do not
  turn pool entries into dataclass instances.
- `FileProposal` stays a `TypedDict`. Only the `ExistingMatch`/`NewEntityProposal` objects it
  holds change shape.
- Cross-file, same-name new-entity deduplication in `persist_new_entities` stays exactly as it
  is today (name-based, via `real_id_by_name`) — this is explicitly out of scope (see spec's
  Non-goals).
- `strategies/sequential/nodes.py`, `strategies/iterative/nodes.py`, and
  `strategies/iterative/state.py` get **no changes** — they treat proposal entity fields
  opaquely.
- There is no pytest (or any test runner) configured for this project (`pyproject.toml` has no
  test dependencies) and adding one is a separate decision, out of scope for this refactor. Every
  task below verifies itself with a small, disposable Python script run directly via the
  project's venv interpreter, using `unittest.mock` (stdlib) where input/output needs to be
  faked. These scripts are written to a `.scratch/` directory at the repo root and deleted
  immediately after use — never committed.
- **Never run the agent against the real, committed `.env`.** It contains real Paperless-ngx
  credentials, `INPUT_FOLDER` pointing at a real folder, and
  `DELETE_INPUT_FILES_AFTER_UPLOAD=true`. `paperlesslabelagent/agent.py` calls
  `load_dotenv(override=True)`, which makes the `.env` file's values override any
  shell-exported environment variables — exporting `INPUT_FOLDER`/`ACCOUNT`/etc. before running
  is **not** sufficient protection. Task 6's manual verification uses a dedicated scratch
  working directory with its own `.env` file for this reason.
- Python interpreter for every command below:
  `"/mnt/c/Users/Frederik/Documents/Arbeitsplatz/git/own/PaperlessLabelAgent/.venv/Scripts/python.exe"`
  (a Windows venv, reachable from this WSL shell — confirmed working; the project is
  editable-installed into it, so `import paperlesslabelagent` works from any working directory).

---

## Task 1: Convert `ExistingMatch`/`NewEntityProposal` to dataclasses

**Files:**
- Modify: `src/paperlesslabelagent/core/state.py:1-27`

**Interfaces:**
- Produces: `ExistingMatch(name: str, confidence: float, reasoning: str, id: int | None = None)` —
  mutable dataclass, `id` defaults to `None`.
- Produces: `NewEntityProposal(entity_type: EntityType, name: str, description: str, reasoning: str, id: int | None = None)` —
  mutable dataclass, `id` defaults to `None`.
- `FileProposal`, `UploadResult`, `AgentState` (rest of the file) are unchanged.

- [ ] **Step 1: Write the failing check**

Create `.scratch/check_task1.py`:

```python
from paperlesslabelagent.core.state import ExistingMatch, NewEntityProposal

match = ExistingMatch(name="Invoice", confidence=0.9, reasoning="looks like an invoice")
assert match.id is None, f"expected id to default to None, got {match.id!r}"
match.id = 5
assert match.id == 5

proposal = NewEntityProposal(entity_type="tag", name="Warranty", description="warranty docs", reasoning="no existing tag fits")
assert proposal.id is None
proposal.id = -1
assert proposal.id == -1

print("OK")
```

- [ ] **Step 2: Run it, verify it fails**

```bash
"/mnt/c/Users/Frederik/Documents/Arbeitsplatz/git/own/PaperlessLabelAgent/.venv/Scripts/python.exe" .scratch/check_task1.py
```

Expected: `AttributeError: 'dict' object has no attribute 'id'` — `ExistingMatch(...)` and
`NewEntityProposal(...)` are currently `TypedDict`s, so calling them just builds a plain dict
with no `id` key (it's `NotRequired` and wasn't passed), and dicts don't support attribute
access.

- [ ] **Step 3: Implement**

Replace `src/paperlesslabelagent/core/state.py` lines 1-27 (the imports through the end of
`NewEntityProposal`) with:

```python
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

EntityType = Literal["tag", "correspondent", "document_type"]


@dataclass
class ExistingMatch:
    """A proposed match against an entity that already exists in the set of tags, correspondents and document types"""

    name: str
    confidence: float
    reasoning: str
    # Set once by check_and_correct_single_proposal, resolved against existing_entities by
    # name - the LLM itself never produces this, it only knows names.
    id: int | None = None


@dataclass
class NewEntityProposal:
    """A proposed new tag, correspondent or document type"""

    entity_type: EntityType
    name: str
    description: str
    reasoning: str
    # Set to a negative placeholder id once folded into existing_entities (see
    # merge_confirmed_new_entities), later replaced with the real Paperless-ngx id by
    # persist_new_entities.
    id: int | None = None
```

Leave the rest of the file (`FileProposal`, `UploadResult`, `AgentState`) untouched — they still
need `TypedDict` from `typing`, which is why the import keeps it alongside `Any` and `Literal`.

- [ ] **Step 4: Run it again, verify it passes**

```bash
"/mnt/c/Users/Frederik/Documents/Arbeitsplatz/git/own/PaperlessLabelAgent/.venv/Scripts/python.exe" .scratch/check_task1.py
```

Expected: `OK`

- [ ] **Step 5: Clean up and commit**

```bash
rm .scratch/check_task1.py
git add src/paperlesslabelagent/core/state.py
git commit -m "refactor: turn ExistingMatch and NewEntityProposal into dataclasses"
```

---

## Task 2: Build proposal objects directly in `classify_document`

**Files:**
- Modify: `src/paperlesslabelagent/core/nodes/classification.py`

**Interfaces:**
- Consumes: `ExistingMatch`, `NewEntityProposal` from Task 1.
- Produces: `classify_document(...)` now returns a `FileProposal` whose
  `proposed_existing_tags`/`proposed_existing_correspondent`/`proposed_existing_document_type`
  are `ExistingMatch` instances (never `.id` set yet — that's Task 3) and whose
  `proposed_new_tags`/`proposed_new_correspondent`/`proposed_new_document_type` are
  `NewEntityProposal` instances (`.id` is `None`).

- [ ] **Step 1: Write the failing check**

Create `.scratch/check_task2.py`:

```python
from types import SimpleNamespace

from paperlesslabelagent.core.nodes import classification
from paperlesslabelagent.core.schemas import ExistingMatchModel, MatchModel, NewEntityProposalModel, build_new_entities_model
from paperlesslabelagent.core.state import ExistingMatch, NewEntityProposal

# Fake the two LLM calls classify_document makes, so this runs with no network/Ollama access.
# Note: classification.matcher/new_entity_proposer are pydantic-based LangChain objects that
# reject arbitrary attribute assignment (`.invoke = ...` raises ValueError: "...has no field...")
# - so replace the whole module-level names instead of patching attributes on the existing objects.
match_result = MatchModel(
    tags=[ExistingMatchModel(name="Invoices", confidence=0.8, reasoning="matches the wording")],
    correspondent=None,
    document_type=None,
)
classification.matcher = SimpleNamespace(invoke=lambda messages: match_result)

new_entities_model = build_new_entities_model(include_tags=False, include_correspondent=True, include_document_type=True)
new_result = new_entities_model(
    new_correspondent=NewEntityProposalModel(entity_type="correspondent", name="Acme", description="a company", reasoning="no existing match"),
    new_document_type=NewEntityProposalModel(entity_type="document_type", name="Invoice", description="an invoice", reasoning="no existing match"),
)
classification.new_entity_proposer = SimpleNamespace(with_structured_output=lambda model: SimpleNamespace(invoke=lambda messages: new_result))

result = classification.classify_document(
    "test.pdf",
    "some invoice text",
    {"tags": {"results": []}, "correspondents": {"results": []}, "document_types": {"results": []}},
    rejected_existing_tags=[],
    rejected_existing_correspondent=None,
    rejected_existing_document_type=None,
    rejected_new_tags=[],
    rejected_new_correspondent=None,
    rejected_new_document_type=None,
)

assert isinstance(result["proposed_existing_tags"][0], ExistingMatch), type(result["proposed_existing_tags"][0])
assert result["proposed_existing_tags"][0].name == "Invoices"
assert isinstance(result["proposed_new_correspondent"], NewEntityProposal), type(result["proposed_new_correspondent"])
assert result["proposed_new_correspondent"].name == "Acme"
assert result["proposed_new_correspondent"].id is None
assert isinstance(result["proposed_new_document_type"], NewEntityProposal)
assert result["proposed_new_document_type"].name == "Invoice"

print("OK")
```

- [ ] **Step 2: Run it, verify it fails**

```bash
"/mnt/c/Users/Frederik/Documents/Arbeitsplatz/git/own/PaperlessLabelAgent/.venv/Scripts/python.exe" .scratch/check_task2.py
```

Expected: `AssertionError` on the first `isinstance` check — `classify_document` currently
builds plain dicts via `.model_dump()`, so `result["proposed_existing_tags"][0]` is a `dict`,
not an `ExistingMatch`.

- [ ] **Step 3: Implement**

In `src/paperlesslabelagent/core/nodes/classification.py`, change the import line:

```python
from paperlesslabelagent.core.state import FileProposal
```

to:

```python
from paperlesslabelagent.core.state import ExistingMatch, FileProposal, NewEntityProposal
```

Replace `format_rejected_existing_entities` and `format_rejected_new_entities` with:

```python
def format_rejected_existing_entities(items: list[ExistingMatch]) -> str:
    """Formats a list of ExistingMatch entries the user already rejected for a document. Returns "" (nothing rendered) if there's nothing to report."""
    if not items:
        return ""
    lines = "\n".join(f'- "{item.name}"' for item in items)
    return f"\nAlready rejected by the user for this document - do not propose these again:\n{lines}\n"


def format_rejected_new_entities(items: list[NewEntityProposal]) -> str:
    """Formats a list of NewEntityProposal entries the user already rejected for a document. Returns "" (nothing rendered) if there's nothing to report."""
    if not items:
        return ""
    lines = "\n".join(f'- "{item.name}" - {item.description}' for item in items)
    return f"\nAlready rejected by the user for this document - do not propose these again:\n{lines}\n"
```

Update `build_user_prompt`'s signature (the `rejected_existing_*` parameters):

```python
def build_user_prompt(
    filename: str,
    text: str,
    existing_entities: dict[str, Any],
    *,
    rejected_existing_tags: list[ExistingMatch] | None = None,
    rejected_existing_correspondent: ExistingMatch | None = None,
    rejected_existing_document_type: ExistingMatch | None = None,
    is_summary: bool = False,
) -> str:
```

Update `build_new_entities_user_prompt`'s signature (the `rejected_new_*` parameters):

```python
def build_new_entities_user_prompt(
    filename: str,
    text: str,
    existing_entities: dict[str, Any],
    *,
    include_tags: bool,
    include_correspondent: bool,
    include_document_type: bool,
    rejected_new_tags: list[NewEntityProposal] | None = None,
    rejected_new_correspondent: NewEntityProposal | None = None,
    rejected_new_document_type: NewEntityProposal | None = None,
    is_summary: bool = False,
) -> str:
```

Update `classify_document`'s signature (the `rejected_*` parameters):

```python
def classify_document(
    filename: str,
    text: str,
    existing_entities: dict[str, Any],
    *,
    rejected_existing_tags: list[ExistingMatch],
    rejected_existing_correspondent: ExistingMatch | None,
    rejected_existing_document_type: ExistingMatch | None,
    rejected_new_tags: list[NewEntityProposal],
    rejected_new_correspondent: NewEntityProposal | None,
    rejected_new_document_type: NewEntityProposal | None,
) -> FileProposal:
```

Replace the `return { ... }` statement at the end of `classify_document` with:

```python
    return {
        "filename": filename,
        "proposed_existing_tags": [ExistingMatch(name=tag.name, confidence=tag.confidence, reasoning=tag.reasoning) for tag in match_result.tags],
        "proposed_existing_correspondent": (
            ExistingMatch(name=match_result.correspondent.name, confidence=match_result.correspondent.confidence, reasoning=match_result.correspondent.reasoning)
            if match_result.correspondent else None
        ),
        "proposed_existing_document_type": (
            ExistingMatch(name=match_result.document_type.name, confidence=match_result.document_type.confidence, reasoning=match_result.document_type.reasoning)
            if match_result.document_type else None
        ),
        "proposed_new_tags": [
            NewEntityProposal(entity_type=proposal.entity_type, name=proposal.name, description=proposal.description, reasoning=proposal.reasoning)
            for proposal in new_tags
        ] or None,
        "proposed_new_correspondent": (
            NewEntityProposal(entity_type=new_correspondent.entity_type, name=new_correspondent.name, description=new_correspondent.description, reasoning=new_correspondent.reasoning)
            if new_correspondent else None
        ),
        "proposed_new_document_type": (
            NewEntityProposal(entity_type=new_document_type.entity_type, name=new_document_type.name, description=new_document_type.description, reasoning=new_document_type.reasoning)
            if new_document_type else None
        ),
        "confirmed": False,
        "needs_retry": False,
        "rejected_existing_tags": rejected_existing_tags,
        "rejected_existing_correspondent": rejected_existing_correspondent,
        "rejected_existing_document_type": rejected_existing_document_type,
        "rejected_new_tags": rejected_new_tags,
        "rejected_new_correspondent": rejected_new_correspondent,
        "rejected_new_document_type": rejected_new_document_type,
    }
```

- [ ] **Step 4: Run it again, verify it passes**

```bash
"/mnt/c/Users/Frederik/Documents/Arbeitsplatz/git/own/PaperlessLabelAgent/.venv/Scripts/python.exe" .scratch/check_task2.py
```

Expected: `OK`

- [ ] **Step 5: Clean up and commit**

```bash
rm .scratch/check_task2.py
git add src/paperlesslabelagent/core/nodes/classification.py
git commit -m "refactor: build ExistingMatch/NewEntityProposal objects directly in classify_document"
```

---

## Task 3: Attribute access in `review.py`

**Files:**
- Modify: `src/paperlesslabelagent/core/nodes/review.py`

**Interfaces:**
- Consumes: `ExistingMatch`, `NewEntityProposal` from Task 1; `FileProposal` entries now hold
  those objects (Task 2).
- Produces: `check_and_correct_single_proposal(...)` now sets `.id` (attribute, not `["id"]`) on
  kept `ExistingMatch` entries, resolved against `existing_entities` by name exactly as before.
  `apply_review_answer(...)` is unchanged (it only moves opaque entries between lists — verified
  by inspection, no attribute/key access on individual entries anywhere in that function).

- [ ] **Step 1: Write the failing check**

Create `.scratch/check_task3.py`:

```python
from unittest.mock import patch

from paperlesslabelagent.core.nodes.review import check_and_correct_single_proposal, collect_hallucination_answer, collect_review_answer, print_proposal
from paperlesslabelagent.core.state import ExistingMatch, NewEntityProposal

# --- check_and_correct_single_proposal: existing-match id resolution by name ---
proposal = {
    "filename": "a.pdf",
    "proposed_existing_tags": [ExistingMatch(name="Invoices", confidence=0.8, reasoning="x")],
    "proposed_existing_correspondent": None,
    "proposed_existing_document_type": None,
    "proposed_new_tags": None,
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
existing_entities = {
    "tags": {"results": [{"id": 7, "name": "Invoices"}]},
    "correspondents": {"results": []},
    "document_types": {"results": []},
}
result = check_and_correct_single_proposal("a.pdf", proposal, existing_entities)
assert result["proposed_existing_tags"][0].id == 7, result["proposed_existing_tags"][0]

# --- print_proposal: must not crash on attribute access ---
print_proposal(("a.pdf", result))

# --- collect_hallucination_answer: reads .name off an ExistingMatch ---
with patch("builtins.input", return_value="y"):
    answer = collect_hallucination_answer({
        "entity": ExistingMatch(name="Warranty", confidence=0.7, reasoning="r"),
        "entity_type": "tag",
        "filename": "a.pdf",
    })
assert answer is True

# --- collect_review_answer: reads attributes off both match kinds ---
review_proposal = {
    "proposed_existing_tags": [ExistingMatch(name="Invoices", confidence=0.8, reasoning="x", id=7)],
    "proposed_new_tags": [NewEntityProposal(entity_type="tag", name="Warranty", description="d", reasoning="r")],
    "proposed_existing_correspondent": None,
    "proposed_new_correspondent": None,
    "proposed_existing_document_type": None,
    "proposed_new_document_type": None,
}
with patch("builtins.input", return_value="y"):
    answer = collect_review_answer({"filename": "a.pdf", "proposal": review_proposal})
assert answer == {"accept_all": True}, answer

print("OK")
```

- [ ] **Step 2: Run it, verify it fails**

```bash
"/mnt/c/Users/Frederik/Documents/Arbeitsplatz/git/own/PaperlessLabelAgent/.venv/Scripts/python.exe" .scratch/check_task3.py
```

Expected: `TypeError: 'ExistingMatch' object is not subscriptable` — `check_and_correct_single_proposal`
still does `tag["name"]`/`tag["id"] = ...` against an `ExistingMatch` dataclass instance, which
doesn't support subscript access.

- [ ] **Step 3: Implement**

In `src/paperlesslabelagent/core/nodes/review.py`, add `NewEntityProposal` to the import:

```python
from paperlesslabelagent.core.state import FileProposal, NewEntityProposal
```

Replace `print_proposal` with:

```python
def print_proposal(proposal):
    print(f"\n --- Proposition of assignments of tags, correspondent and document types === {proposal[0]} === as well as new ones tags, correspondents and document types ---")

    if proposal[1]["proposed_existing_tags"]:
        for tag in proposal[1]["proposed_existing_tags"]:
            print(f'  Tag: "{tag.name}" (confidence={tag.confidence:.2f})')
    else:
        print("  Tags: (no match)")

    correspondent = proposal[1]["proposed_existing_correspondent"]
    if correspondent:
        print(f'  Correspondent: "{correspondent.name}" (confidence={correspondent.confidence:.2f})')
    else:
        print("  Correspondent: (no match)")

    document_type = proposal[1]["proposed_existing_document_type"]
    if document_type:
        print(f'  Document type: "{document_type.name}" (confidence={document_type.confidence:.2f})')
    else:
        print("  Document type: (no match)")

    for new_tag in proposal[1]["proposed_new_tags"] or []:
        print(f'  New tag proposed: "{new_tag.name}" - {new_tag.description}')

    new_correspondent = proposal[1]["proposed_new_correspondent"]
    if new_correspondent:
        print(f'  New correspondent proposed: "{new_correspondent.name}" - {new_correspondent.description}')
    else:
        print("  New correspondent: (no match)")

    new_document_type = proposal[1]["proposed_new_document_type"]
    if new_document_type:
        print(f'  New document type proposed: "{new_document_type.name}" - {new_document_type.description}')
    else:
        print("  New document type: (no match)")
```

Replace the body of `check_and_correct_single_proposal` (keep its signature) with:

```python
def check_and_correct_single_proposal(filename: str, proposal: FileProposal, existing_entities: dict[str, Any]) -> FileProposal:
    """Checks a single proposal for erroneous assignments of hallucinated existing entries.

    For every tag/correspondent/document_type the LLM matched that doesn't actually exist
    in existing_entities (name not found), and asks whether to
    keep it as a new-entity proposal instead, or reject it and retry.
    """

    tags = existing_entities.get("tags", {}).get("results", [])
    correspondents = existing_entities.get("correspondents", {}).get("results", [])
    document_types = existing_entities.get("document_types", {}).get("results", [])

    tag_ids_by_name = {item["name"]: item["id"] for item in tags}
    correspondent_ids_by_name = {item["name"]: item["id"] for item in correspondents}
    document_type_ids_by_name = {item["name"]: item["id"] for item in document_types}

    kept_tags = []
    for tag in proposal["proposed_existing_tags"]:
        if tag.name in tag_ids_by_name:
            tag.id = tag_ids_by_name[tag.name]
            kept_tags.append(tag)
            continue

        # Not a existing tag - the LLM hallucinated it. Use it as a new proposed entity instead of existing one.
        keep_as_new = interrupt({"kind": "hallucination", "entity_type": "tag", "filename": filename, "entity": tag})
        new_tag_proposal = NewEntityProposal(entity_type="tag", name=tag.name, description=tag.reasoning, reasoning=tag.reasoning)
        if keep_as_new:
            new_tags = proposal.get("proposed_new_tags") or []
            new_tags.append(new_tag_proposal)
            proposal["proposed_new_tags"] = new_tags
        else:
            rejected_new_tags = proposal.get("rejected_new_tags") or []
            rejected_new_tags.append(new_tag_proposal)
            proposal["rejected_new_tags"] = rejected_new_tags
            proposal["needs_retry"] = True
    proposal["proposed_existing_tags"] = kept_tags

    for proposed_existing_entity_type, entity_type, new_entity_type, rejected_new_entity_type, ids_by_name in (
        ("proposed_existing_correspondent", "correspondent", "proposed_new_correspondent", "rejected_new_correspondent", correspondent_ids_by_name),
        ("proposed_existing_document_type", "document_type", "proposed_new_document_type", "rejected_new_document_type", document_type_ids_by_name),
    ):
        value = proposal[proposed_existing_entity_type]
        if value is None:
            continue
        if value.name in ids_by_name:
            value.id = ids_by_name[value.name]
            continue

        # Not a existing correspondent or document_type, hallucinated by LLM --- see comment above.
        keep_as_new = interrupt({"kind": "hallucination", "entity_type": entity_type, "filename": filename, "entity": value})
        new_entity_proposal = NewEntityProposal(entity_type=entity_type, name=value.name, description=value.reasoning, reasoning=value.reasoning)
        if keep_as_new:
            proposal[new_entity_type] = new_entity_proposal
        else:
            proposal[rejected_new_entity_type] = new_entity_proposal
            proposal["needs_retry"] = True
        proposal[proposed_existing_entity_type] = None

    return proposal
```

Leave `apply_review_answer` completely unchanged.

Replace `collect_hallucination_answer` with:

```python
def collect_hallucination_answer(hallucinated_proposal: dict[str, Any]) -> bool:
    """Turns a 'hallucination' interrupt into a y/n question for the user."""
    entity = hallucinated_proposal["entity"]
    entity_type = hallucinated_proposal["entity_type"]
    print(f'\nProposal for "{hallucinated_proposal["filename"]}" used a {entity_type} not found in Paperless-ngx (likely a hallucination): "{entity.name}".')
    return ask_yes_no("Treat it as a new entity instead of rejecting it?")
```

Replace `collect_review_answer` with:

```python
def collect_review_answer(proposal_to_verify: dict[str, Any]) -> dict[str, Any]:
    """Turns a 'verify' interrupt into the full accept/reject Q&A for one file's proposal"""
    filename = proposal_to_verify["filename"]
    proposal = proposal_to_verify["proposal"]
    print_proposal((filename, proposal))

    if ask_yes_no("\nDo you fully accept the proposed assignments and new entities?"):
        return {"accept_all": True}

    answer: dict[str, Any] = {"accept_all": False}

    answer["proposed_existing_tags"] = [
        ask_yes_no(f'  Keep tag "{tag.name}" (confidence={tag.confidence:.2f})?') for tag in proposal["proposed_existing_tags"]
    ]
    answer["proposed_new_tags"] = [
        ask_yes_no(f'  Add new tag "{tag.name}" - {tag.description}?') for tag in proposal.get("proposed_new_tags") or []
    ]

    for proposal_key, label in (
        ("proposed_existing_correspondent", "correspondent"),
        ("proposed_new_correspondent", "new correspondent"),
        ("proposed_existing_document_type", "document type"),
        ("proposed_new_document_type", "new document type"),
    ):
        value = proposal.get(proposal_key)
        answer[proposal_key] = ask_yes_no(f'  Accept {label} "{value.name}"?') if value else None

    return answer
```

- [ ] **Step 4: Run it again, verify it passes**

```bash
"/mnt/c/Users/Frederik/Documents/Arbeitsplatz/git/own/PaperlessLabelAgent/.venv/Scripts/python.exe" .scratch/check_task3.py
```

Expected: `OK` (plus the `print_proposal` output printed above it — that's fine, it's meant to
print).

- [ ] **Step 5: Clean up and commit**

```bash
rm .scratch/check_task3.py
git add src/paperlesslabelagent/core/nodes/review.py
git commit -m "refactor: switch review.py to attribute access on proposal entities"
```

---

## Task 4: Mutate-in-place placeholder ids in `entities.py`

**Files:**
- Modify: `src/paperlesslabelagent/core/nodes/entities.py:53-83`

**Interfaces:**
- Consumes: `NewEntityProposal` from Task 1.
- Produces: `_add_entity_to_pool(...)` now mutates the `entity` argument's `.id` in place (instead
  of returning/appending a copy) — so the same `NewEntityProposal` object referenced from a
  `FileProposal`'s `proposed_new_tags`/`proposed_new_correspondent`/`proposed_new_document_type`
  already has its placeholder id set by the time `merge_confirmed_new_entities` returns, with no
  separate propagation step needed for this stage (unlike the *real* id, resolved later in
  Task 5).

- [ ] **Step 1: Write the failing check**

Create `.scratch/check_task4.py`:

```python
from paperlesslabelagent.core.nodes.entities import _add_entity_to_pool, merge_confirmed_new_entities
from paperlesslabelagent.core.state import NewEntityProposal

entity = NewEntityProposal(entity_type="tag", name="Invoices", description="d", reasoning="r")
existing_entities: dict = {}
confirmed_new_entities: list = []
_add_entity_to_pool(existing_entities, confirmed_new_entities, "tag", entity)

assert entity.id == -1, entity.id
assert confirmed_new_entities[0] is entity
assert existing_entities["tags"]["results"][0] == {"id": -1, "name": "Invoices"}

# merge_confirmed_new_entities: the FileProposal's own object must see the placeholder id.
proposal_entity = NewEntityProposal(entity_type="correspondent", name="Acme", description="d", reasoning="r")
proposal = {
    "proposed_new_tags": None,
    "proposed_new_correspondent": proposal_entity,
    "proposed_new_document_type": None,
}
new_existing_entities, new_confirmed = merge_confirmed_new_entities({}, proposal, [])
assert proposal["proposed_new_correspondent"].id == -1, proposal["proposed_new_correspondent"].id
assert proposal_entity.id == -1

print("OK")
```

- [ ] **Step 2: Run it, verify it fails**

```bash
"/mnt/c/Users/Frederik/Documents/Arbeitsplatz/git/own/PaperlessLabelAgent/.venv/Scripts/python.exe" .scratch/check_task4.py
```

Expected: `TypeError: 'NewEntityProposal' object is not subscriptable` — the current
`_add_entity_to_pool` does `entity["name"]` (and later `{**entity, "id": temp_id}`) against a
dataclass instance, neither of which a plain dataclass supports.

- [ ] **Step 3: Implement**

Replace `_add_entity_to_pool` in `src/paperlesslabelagent/core/nodes/entities.py` with:

```python
def _add_entity_to_pool(
    existing_entities: dict[str, Any],
    confirmed_new_entities: list[NewEntityProposal],
    entity_type: EntityType,
    entity: NewEntityProposal,
) -> None:
    temp_id = -(len(confirmed_new_entities) + 1)
    category_key = f"{entity_type}s"
    entity.id = temp_id
    existing_entities.setdefault(category_key, {}).setdefault("results", []).append({"id": temp_id, "name": entity.name})
    confirmed_new_entities.append(entity)
```

`merge_confirmed_new_entities` needs no changes — it already passes the proposal's own entity
objects (`proposal.get("proposed_new_tags")` entries, `proposal.get("proposed_new_correspondent")`,
`proposal.get("proposed_new_document_type")`) straight into `_add_entity_to_pool`.

- [ ] **Step 4: Run it again, verify it passes**

```bash
"/mnt/c/Users/Frederik/Documents/Arbeitsplatz/git/own/PaperlessLabelAgent/.venv/Scripts/python.exe" .scratch/check_task4.py
```

Expected: `OK`

- [ ] **Step 5: Clean up and commit**

```bash
rm .scratch/check_task4.py
git add src/paperlesslabelagent/core/nodes/entities.py
git commit -m "refactor: mutate new-entity placeholder ids in place instead of copying"
```

---

## Task 5: Propagate real ids onto proposals; simplify `persist_file_proposals`

**Files:**
- Modify: `src/paperlesslabelagent/core/nodes/resultpersistence.py`

**Interfaces:**
- Consumes: `NewEntityProposal`, `FileProposal` from Task 1; placeholder ids already set in
  place on proposal objects (Task 4).
- Produces: `persist_new_entities(...)` now also returns an updated `"proposals"` key in which
  every confirmed proposal's new-entity objects have `.id` set to the real, server-assigned id
  (not just `confirmed_new_entities`, as before). `persist_file_proposals(...)` reads `.id`
  directly off each proposal's tag/correspondent/document_type objects — no more
  `*_ids_by_name` dict rebuilding.

- [ ] **Step 1: Write the failing check**

Create `.scratch/check_task5.py`:

```python
import os

os.environ["ACCOUNT"] = "mock"
os.environ["PASSWORD"] = "mock"
os.environ.pop("DELETE_INPUT_FILES_AFTER_UPLOAD", None)

from paperlesslabelagent.core.nodes.resultpersistence import persist_file_proposals, persist_new_entities
from paperlesslabelagent.core.state import ExistingMatch, NewEntityProposal

new_tag = NewEntityProposal(entity_type="tag", name="Invoices", description="d", reasoning="r")
new_tag.id = -1  # placeholder, as if _add_entity_to_pool already ran (Task 4)

proposal = {
    "filename": "a.pdf",
    "proposed_existing_tags": [ExistingMatch(name="Existing", confidence=0.9, reasoning="r", id=3)],
    "proposed_existing_correspondent": None,
    "proposed_existing_document_type": None,
    "proposed_new_tags": [new_tag],
    "proposed_new_correspondent": None,
    "proposed_new_document_type": None,
    "confirmed": True,
    "needs_retry": False,
    "rejected_existing_tags": [],
    "rejected_existing_correspondent": None,
    "rejected_existing_document_type": None,
    "rejected_new_tags": [],
    "rejected_new_correspondent": None,
    "rejected_new_document_type": None,
}

state = {
    "input_folder": "/nonexistent",
    "existingEntities": {
        "tags": {"results": [{"id": 3, "name": "Existing"}, {"id": -1, "name": "Invoices"}]},
        "correspondents": {"results": []},
        "document_types": {"results": []},
    },
    "file_texts": {},
    "proposals": {"a.pdf": proposal},
    "pending_new_entities": [],
    "confirmed_new_entities": [new_tag],
    "created_entity_ids": {},
}

result1 = persist_new_entities(state)
propagated = result1["proposals"]["a.pdf"]["proposed_new_tags"][0]
assert propagated.id == 4, propagated.id  # real_ids=[3] -> mock-created id is 3+1=4

state.update(result1)
result2 = persist_file_proposals(state)
assert result2["upload_results"]["a.pdf"]["task_id"] == "mock-task-a.pdf", result2

print("OK")
```

- [ ] **Step 2: Run it, verify it fails**

```bash
"/mnt/c/Users/Frederik/Documents/Arbeitsplatz/git/own/PaperlessLabelAgent/.venv/Scripts/python.exe" .scratch/check_task5.py
```

Expected: `TypeError: 'NewEntityProposal' object is not iterable` — the current
`persist_new_entities` does `confirmed_new_entities = [dict(entity) for entity in ...]`, and
`dict()` can't convert an arbitrary object (only a mapping or an iterable of key-value pairs),
which a plain dataclass instance is neither.

- [ ] **Step 3: Implement**

In `src/paperlesslabelagent/core/nodes/resultpersistence.py`, change the imports at the top:

```python
import os
from typing import Any

from paperlesslabelagent.core.state import AgentState
from paperlesslabelagent.core.nodes.paperlesstools import (
    TAGS_API_NAME, CORRESPONDENTS_API_NAME, DOCUMENT_TYPES_API_NAME,
    create_tag, create_correspondent, create_document_type, uses_mock_entities, upload_document,
)
```

to:

```python
import os
from dataclasses import replace
from typing import Any

from paperlesslabelagent.core.state import AgentState, FileProposal, NewEntityProposal
from paperlesslabelagent.core.nodes.paperlesstools import (
    TAGS_API_NAME, CORRESPONDENTS_API_NAME, DOCUMENT_TYPES_API_NAME,
    create_tag, create_correspondent, create_document_type, uses_mock_entities, upload_document,
)
```

Add this helper directly above `persist_new_entities`:

```python
def _propagate_real_id(new_entity: NewEntityProposal | None, real_id_by_placeholder: dict[int, int]) -> None:
    """Writes a new entity's real, server-assigned id onto its own proposal object once
    persist_new_entities has resolved it. This can't rely on shared object identity with
    confirmed_new_entities - LangGraph checkpoints each state key independently, so by the
    time this node runs, the two are already separate copies even if they started out as the
    same object when merge_confirmed_new_entities (core/nodes/entities.py) first set it."""
    if new_entity is not None and new_entity.id in real_id_by_placeholder:
        new_entity.id = real_id_by_placeholder[new_entity.id]
```

Replace the body of `persist_new_entities` (keep its signature and docstring) with:

```python
def persist_new_entities(state: AgentState) -> dict[str, Any]:
    """Creates every confirmed new entity (negative placeholder id in existingEntities) in
    Paperless-ngx for real, replacing its placeholder id with the real, server-assigned one.
    """
    ACCOUNT, PASSWORD, API_URL = os.getenv("ACCOUNT"), os.getenv("PASSWORD"), os.getenv("API_URL")
    is_mock = uses_mock_entities(ACCOUNT, PASSWORD)

    existing_entities = {key: {**value, "results": list(value.get("results", []))} for key, value in state["existingEntities"].items()}
    confirmed_new_entities = [replace(entity) for entity in state.get("confirmed_new_entities", [])]
    real_id_by_placeholder: dict[int, int] = {}

    for category_key, create in ENTITY_API_NAMES.items():
        results = existing_entities.get(category_key, {}).get("results", [])
        # Two different files' proposals can each add the same-named new entity as its own
        # placeholder (merge_confirmed_new_entities does not dedupe by name), so track ids by
        # name as we go and only actually create one Paperless-ngx entity per distinct name.
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
        if entity.id in real_id_by_placeholder:
            entity.id = real_id_by_placeholder[entity.id]

    proposals: dict[str, FileProposal] = dict(state.get("proposals", {}))
    for proposal in proposals.values():
        for tag in proposal.get("proposed_new_tags") or []:
            _propagate_real_id(tag, real_id_by_placeholder)
        _propagate_real_id(proposal.get("proposed_new_correspondent"), real_id_by_placeholder)
        _propagate_real_id(proposal.get("proposed_new_document_type"), real_id_by_placeholder)

    return {"existingEntities": existing_entities, "confirmed_new_entities": confirmed_new_entities, "proposals": proposals}
```

Replace the body of `persist_file_proposals` (keep its signature and docstring) with:

```python
def persist_file_proposals(state: AgentState) -> dict[str, Any]:
    """ Persists the file proposals to Paperless-ngx by uploading the files and assigning the tags, correspondents and document types."""
    ACCOUNT, PASSWORD, API_URL = os.getenv("ACCOUNT"), os.getenv("PASSWORD"), os.getenv("API_URL")
    is_mock = uses_mock_entities(ACCOUNT, PASSWORD)
    delete_after_upload = (os.getenv("DELETE_INPUT_FILES_AFTER_UPLOAD") or "").strip().lower() == "true"

    upload_results = dict(state.get("upload_results", {}))

    for filename, proposal in state.get("proposals", {}).items():
        if not proposal.get("confirmed") or filename in upload_results:
            continue  # skip unconfirmed proposals (e.g. retry limit hit) and files already uploaded (replay safety)

        tags = list(proposal.get("proposed_existing_tags") or []) + list(proposal.get("proposed_new_tags") or [])
        tag_ids = [tag.id for tag in tags]

        correspondent = proposal.get("proposed_existing_correspondent") or proposal.get("proposed_new_correspondent")
        correspondent_id = correspondent.id if correspondent else None

        document_type = proposal.get("proposed_existing_document_type") or proposal.get("proposed_new_document_type")
        document_type_id = document_type.id if document_type else None

        file_path = os.path.join(state["input_folder"], filename)
        upload_results[filename] = _simulate_upload(filename) if is_mock \
            else upload_document(file_path, API_URL, ACCOUNT, PASSWORD, tag_ids, correspondent_id, document_type_id)

        if delete_after_upload:
            os.remove(file_path)

    return {"upload_results": upload_results}
```

- [ ] **Step 4: Run it again, verify it passes**

```bash
"/mnt/c/Users/Frederik/Documents/Arbeitsplatz/git/own/PaperlessLabelAgent/.venv/Scripts/python.exe" .scratch/check_task5.py
```

Expected: `OK`

- [ ] **Step 5: Clean up and commit**

```bash
rm .scratch/check_task5.py
git add src/paperlesslabelagent/core/nodes/resultpersistence.py
git commit -m "refactor: propagate real new-entity ids onto proposals; read .id directly on persist"
```

---

## Task 6: Manual end-to-end verification (both strategies, mock mode)

**Files:** none (verification only, no commit)

This exercises the full graph — including the `interrupt()`-driven review flow and the real
LLM calls that Tasks 1-5's isolated checks intentionally avoided — end to end. It requires a
locally running Ollama with a structured-output-capable model pulled, and Tesseract OCR
installed, neither of which is available in this sandbox (`ollama` was not found here). **This
task must be run by a human on a machine that has both**, not executed unattended by an
implementing agent. If you're executing this plan and don't have Ollama/Tesseract available,
stop here, report Tasks 1-5 complete, and ask the user to run this task and report back before
merging.

- [ ] **Step 1: Set up an isolated scratch environment**

Do this **outside** the repository and **outside** the real `INPUT_FOLDER`. Do not reuse or
point at the real, committed `.env` — `agent.py` calls `load_dotenv(override=True)`, so if the
real `.env` is anywhere it can find it, its `INPUT_FOLDER` and `DELETE_INPUT_FILES_AFTER_UPLOAD=true`
would silently win over anything exported in the shell.

```bash
mkdir -p /mnt/c/Users/Frederik/paperless-verify/input
```

Copy one or two real PDF files you don't mind testing with into
`/mnt/c/Users/Frederik/paperless-verify/input/` — ideally at least one that will need a brand
new tag, correspondent, and document type (nothing in `test/paperless-instance-mock/*` should
match it well).

Create `/mnt/c/Users/Frederik/paperless-verify/.env`:

```
ACCOUNT=mock
PASSWORD=mock
API_URL=
MODEL=<a model tag available in your local `ollama list`, supporting structured output>
TESSDATA_PATH=<your local tessdata path>
OCR_LANGUAGES=eng
INPUT_FOLDER=/mnt/c/Users/Frederik/paperless-verify/input
STRATEGY=iterative
ENTITY_LANGUAGE=english
DELETE_INPUT_FILES_AFTER_UPLOAD=false
```

- [ ] **Step 2: Run the iterative strategy**

```bash
cd /mnt/c/Users/Frederik/paperless-verify
"/mnt/c/Users/Frederik/Documents/Arbeitsplatz/git/own/PaperlessLabelAgent/.venv/Scripts/python.exe" -m paperlesslabelagent.agent
```

Walk through the `[y/n]` prompts. For the file expected to need new entities, accept the new
tag/correspondent/document type proposals when asked. Confirm:
- No `AttributeError`/`KeyError` crash at any point, especially not during the final upload step
  (that would mean a proposal reached `persist_file_proposals` with a `None`/stale `.id`).
- The run ends with `Done — all proposals confirmed.` (or, if you deliberately rejected
  something down to the retry limit, the matching "N file(s) hit the retry limit" message — not
  a stack trace).

- [ ] **Step 3: Run the sequential strategy**

Edit `/mnt/c/Users/Frederik/paperless-verify/.env`, change `STRATEGY=iterative` to
`STRATEGY=sequential`. If you have two input files that could plausibly each propose the same
new tag/correspondent name independently, this is the strategy where that matters (see the
plan's Global Constraints — that dedup logic is unchanged, this just confirms it still works
against the new dataclass-based proposals). Re-run:

```bash
cd /mnt/c/Users/Frederik/paperless-verify
"/mnt/c/Users/Frederik/Documents/Arbeitsplatz/git/own/PaperlessLabelAgent/.venv/Scripts/python.exe" -m paperlesslabelagent.agent
```

Confirm the same things as Step 2.

- [ ] **Step 4: Clean up**

```bash
rm -rf /mnt/c/Users/Frederik/paperless-verify
```

No commit for this task — it's verification only, confirming Tasks 1-5's changes work together
against the real interactive/LLM flow.
