# Entity ID handling: classes instead of a scattered name→id mapping table

## Problem

Entity IDs (for tags, correspondents, document types) are currently threaded through the
agent as plain `TypedDict`s (`ExistingMatch`, `NewEntityProposal`) that only sometimes carry
an `id`. Whenever a consumer needs an entity's id, it rebuilds a `name → id` dict from
`existingEntities` on the spot:

- `core/nodes/review.py::check_and_correct_single_proposal` builds three such dicts.
- `core/nodes/resultpersistence.py::persist_file_proposals` builds three more.
- `core/nodes/resultpersistence.py::persist_new_entities` builds `real_id_by_name` /
  `real_id_by_placeholder`, but only patches `confirmed_new_entities` — never the original
  `FileProposal` objects that live in `state["proposals"]`.

That last point is the concrete bug: once `persist_new_entities` replaces a new entity's
negative placeholder id with its real Paperless-ngx id, `state["proposals"][filename]`'s own
copy of that entity never learns the real id. `persist_file_proposals` works around this by
re-deriving ids by name from `existingEntities` yet again, rather than trusting the id already
attached to the proposal.

`core/nodes/resultpersistence.py:48` already flags this with a TODO: *"we could use objects
instead of plain dicts."*

## Goal

Make `id` resolution happen in one clearly-identifiable place per lifecycle stage, and have
every consumer trust `entity.id` once it's set, instead of re-deriving it from
`existingEntities` by name.

## Non-goals

- Deduplicating same-named new-entity proposals made independently by different files within
  the same sequential batch (e.g. two files each proposing a new tag "Invoice" before either
  is reviewed). This is already only reconciled by name at persist time
  (`persist_new_entities`), and stays that way. This case cannot arise in the iterative
  strategy, since `review_current_proposal` folds an accepted new entity into `existingEntities`
  before the next document is classified, so later documents match it by name like any other
  existing entity.
- Changing the shape of the `existingEntities` pool itself. It mirrors the Paperless-ngx API
  response shape (list of `{"id": ..., "name": ..., ...}` dicts per category) and stays that
  way; only the AI's own proposal objects change.

## Design

### Data model

`ExistingMatch` and `NewEntityProposal` (`core/state.py`) become `@dataclass` instead of
`TypedDict`, each with a mutable `id: int | None = None` field (replacing today's
`NotRequired[int]`). `FileProposal` stays a `TypedDict`; only the entity objects it holds
change from dict literals to dataclass instances.

### Why not shared object references

The natural-looking design — give a new entity's placeholder object to every place that
references it, then mutate its `.id` once and have every reference see the update — does not
work reliably here. Both `sequential` and `iterative` graphs use `InMemorySaver` with
`interrupt()`-driven human review on essentially every document, so the graph checkpoints
constantly. `InMemorySaver` stores each top-level state key (`proposals`,
`confirmed_new_entities`, `existingEntities`, ...) as an **independently serialized blob**. Two
objects that are identical (same instance) at the moment a node returns are, after the next
checkpoint/resume, two separate deserialized copies with equal values but no shared identity.
So any design relying on "mutate once, every reference sees it" would appear to work within a
single node call and then silently stop working the moment a checkpoint boundary is crossed —
which happens constantly given the interrupt-heavy flow.

Consequently, propagation of a resolved id onto every `FileProposal` that references an entity
must be an **explicit step**, not an incidental effect of shared references.

### ID resolution, in two stages

**Existing entities** (already in Paperless-ngx): `check_and_correct_single_proposal` is
already the single place that resolves a matched name against the `existingEntities` pool. It
keeps doing exactly that, just via `tag.id = ...` instead of `tag["id"] = ...`. This id is
*not* guaranteed to be final, though: the `existingEntities` pool it resolves against can
itself already contain a not-yet-persisted new entity from an earlier proposal in the same run
(folded in by `merge_confirmed_new_entities`, with a negative placeholder id — see
"New entities" below). If a later document's match names that same entity,
`check_and_correct_single_proposal` correctly resolves `tag.id` to whatever is in the pool at
that moment, which is still the placeholder. So an `ExistingMatch` object can carry a negative
placeholder id too, not just a `NewEntityProposal` — and it needs the same real-id propagation
pass described below.

**New entities**: two moments matter.

1. *Placeholder assignment* (`core/nodes/entities.py::_add_entity_to_pool`, called via
   `merge_confirmed_new_entities` from `review_current_proposal` / `user_verify_proposals`):
   today this builds a **copy** of the entity (`{**entity, "id": temp_id}`) for
   `confirmed_new_entities`, leaving the original object referenced from
   `proposal["proposed_new_tags"]` etc. untouched. Change this to mutate the entity object
   that's passed in (`entity.id = temp_id`) instead of copying. This happens inside the same
   node call that owns the proposal, before any checkpoint boundary, so the placeholder id
   ends up correctly set on both the pool and the proposal's own object for free.

2. *Real-id resolution* (`core/nodes/resultpersistence.py::persist_new_entities`): this already
   computes `real_id_by_placeholder: dict[int, int]` by deduping on name against the
   `existingEntities` pool — that logic is unchanged (see Non-goals). New: an explicit
   propagation pass walks **all six** of every confirmed proposal's tag/correspondent/
   document_type fields — both the `proposed_existing_*` ones (per the correction above) and
   the `proposed_new_*` ones — and, for any entity object whose `.id` is a key in
   `real_id_by_placeholder`, sets `.id` to the resolved real id. Correlating by placeholder id
   (a plain `int`) rather than by object identity is what makes this safe across checkpoint
   boundaries — `persist_new_entities` runs as a separate node, well after the proposal and
   the entity pool have already been through at least one independent checkpoint/resume cycle
   by the time it starts. Missing the `proposed_existing_*` fields here was exactly the
   Critical bug caught in final review: it let an `ExistingMatch`'s placeholder id survive
   all the way to `persist_file_proposals` and get shipped to `upload_document`.

### Consumer simplification

Because both stages above guarantee every confirmed proposal's tag/correspondent/document_type
objects carry a resolved `.id` by the time `persist_file_proposals` runs (both graphs run
`persist_new_entities` → `persist_file_proposals` in that order), `persist_file_proposals` no
longer needs to rebuild `tag_ids_by_name` / `correspondent_ids_by_name` /
`document_type_ids_by_name` from `existingEntities`. It reads `.id` directly off each proposal's
entity objects instead.

## Files touched

- `core/state.py` — `ExistingMatch`, `NewEntityProposal` → `@dataclass` with mutable
  `id: int | None = None`. `FileProposal` / `AgentState` unchanged in shape.
- `core/nodes/classification.py` — `classify_document` constructs `ExistingMatch(...)` /
  `NewEntityProposal(...)` directly from the LLM's pydantic `ExistingMatchModel` /
  `NewEntityProposalModel` fields, instead of `.model_dump()` into plain dicts.
  `format_rejected_existing_entities` / `format_rejected_new_entities` switch to attribute
  access.
- `core/nodes/review.py` — `print_proposal`, `check_and_correct_single_proposal`,
  `collect_hallucination_answer`, `collect_review_answer` switch bracket access to attribute
  access; ad hoc `{...}` dict literals for new-entity proposals become `NewEntityProposal(...)`
  construction. `apply_review_answer` is untouched — it only shuffles opaque entries between
  lists.
- `core/nodes/entities.py` — `_add_entity_to_pool` mutates the passed-in entity's `.id` in
  place instead of spreading into a copy. Pool entries in `existingEntities[...]["results"]`
  stay plain `{"id": ..., "name": ...}` dicts — that pool mirrors Paperless-ngx API shape and
  has no reason to change.
- `core/nodes/resultpersistence.py` — `persist_new_entities` gains the propagation pass
  described above. `persist_file_proposals` drops its three `*_ids_by_name` dict
  comprehensions and reads `.id` directly.
- `strategies/sequential/nodes.py`, `strategies/iterative/nodes.py`,
  `strategies/iterative/state.py` — **no changes**; they treat proposal entity fields opaquely
  and never reach into `ExistingMatch` / `NewEntityProposal` internals.

## Testing

There's no existing automated test suite for this project (only vendored packages under
`.venv`). The mock-mode path (`ACCOUNT`/`PASSWORD` containing `"mock"`) already exercises the
full flow end-to-end against `test/paperless-instance-mock/*` without a real Paperless-ngx
instance, and is the intended way to verify this refactor.

**Do not use the real configured input folder for this.** If `DELETE_INPUT_FILES_AFTER_UPLOAD`
is enabled in the active `.env` and something goes wrong, files would be deleted. Verification
should run both strategies (`sequential` and `iterative`) in mock mode against a throwaway
input folder (outside the repo's real input folder) containing at least one file that needs a
brand-new tag, a new correspondent, and a new document type — confirming the upload step ends
up with correct, positive, mock-simulated ids and no leftover negative placeholders or
`AttributeError`/`KeyError` from a missed `.id`.

Whether to add real automated tests (there's currently no pytest setup) is a separate decision,
not part of this refactor.
