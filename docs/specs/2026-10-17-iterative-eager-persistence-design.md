# Persisting confirmed iterative-run work immediately, so an abort neither loses nor redoes it

## Problem

The iterative strategy (`strategies/iterative/graph.py`) only pushes results to Paperless-ngx
once, after the entire document loop finishes: `review_current_proposal` loops back to
`classify_current_document` for as long as `select_next_document` finds work, and only routes
to `persist_new_entities` → `persist_file_proposals` once nothing is left. If the agent is
aborted (crash, killed process, closed terminal) before that point, every classification
confirmed so far is lost — nothing was ever written to Paperless-ngx.

Fixing only that (uploading each document as soon as it is confirmed) is not enough on its own.
With `DELETE_INPUT_FILES_AFTER_UPLOAD=false` (the documented default), an uploaded file stays in
the input folder. A restarted run starts with empty state, finds the file again, and would
reclassify, re-review and re-upload it, producing a duplicate document in Paperless-ngx.

## Goal

- A confirmed document's classification is applied to Paperless-ngx immediately, not batched to
  the end of a run.
- Restarting the agent after an abort neither reclassifies, re-reviews nor re-uploads documents
  that a previous run already confirmed.

## Non-goals

- Resuming the agent's own in-flight state across a process restart (e.g. a durable
  `SqliteSaver` checkpointer). An earlier draft of this design proposed that; it was dropped
  because the input folder itself becomes the record of remaining work (see "Why no durable
  checkpointing" below). The graph keeps its `InMemorySaver`, which is still required for the
  `interrupt()`-based review within a single run.
- Rewiring the sequential strategy's graph (`strategies/sequential`). It does pick up the
  shared file-moving behavior below, since that lives in `core`.
- A configurable name or location for the processed folder. It is fixed to
  `<INPUT_FOLDER>/processed` until there is a need for more.
- Retrying failed Paperless-ngx API calls.

## Design

### Per-document eager persistence (iterative graph)

Move `persist_new_entities` → `persist_file_proposals` inside the per-document loop, running
immediately after every `review_current_proposal`, instead of once after the whole loop exits.
The `select_next_document`-based routing (more documents → `classify_current_document`, none →
`END`) moves from directly after `review_current_proposal` to after `persist_file_proposals`.

This is safe per document because `merge_confirmed_new_entities` (already called from
`review_current_proposal` on confirmation) folds a confirmed document's new entities into
`existingEntities`, with placeholder ids, before that node returns. A later document's
classification therefore matches against the earlier one's entity as an existing entity rather
than proposing it again, so `persist_new_entities` only ever has to resolve the current
document's placeholders. Its dedupe-by-name logic stays as a safety net.

Both persist nodes also run after iterations that end in a retry rather than a confirmation.
They are no-ops there: `persist_file_proposals` skips proposals whose `confirmed` is falsy, and
`persist_new_entities` only finds placeholder ids that `merge_confirmed_new_entities` added,
which happens only on confirmation.

Trade-off: nothing in the batch stays off live Paperless-ngx until the whole batch is reviewed.
A confirmed document is live immediately, with no in-agent undo; fixing a mistaken
confirmation means editing it in Paperless-ngx. This is intentional, in favor of durability.

### Moving processed files out of the input folder

In `persist_file_proposals` (`core/nodes/resultpersistence.py`), after a file's upload succeeds:

- `DELETE_INPUT_FILES_AFTER_UPLOAD=true` → delete the file, as today.
- otherwise → move it to `<INPUT_FOLDER>/processed/`, creating that folder if it does not exist.
  If a file of the same name already exists there, append a numeric suffix before the extension
  (`invoice.pdf` → `invoice (1).pdf`, then `invoice (2).pdf`, …) so an earlier original is never
  overwritten.

The input folder then always contains exactly the work that remains, which is what lets a
restarted run skip already-confirmed documents without any persisted agent state.

This applies in mock mode as well, matching today's behavior where deletion also runs in mock
mode. Because the logic is in `core`, the sequential strategy moves its files too, after its
end-of-batch upload.

### Ignoring subfolders when loading documents

`load_documents` (`core/nodes/entities.py`) currently passes every entry of
`os.listdir(input_folder)` to `extractTextFromPDF`. Once `processed/` exists this would fail on
the next run. It must only consider regular files (`os.path.isfile`) and skip directories.

### Failures while persisting

A failed entity creation or upload raises, as it does today, and stops the run. With eager
persistence this is safe: every previously confirmed document is already in Paperless-ngx and
out of the input folder, so a restart redoes only the failing document. Entities created for
that document before its upload failed already exist in Paperless-ngx, so
`fetch_existing_entities` picks them up and the restarted classification matches them as
existing entities.

### Why no durable checkpointing

A durable checkpointer would additionally preserve, across a restart:

- the one document that was mid-classification or mid-review when the abort happened, and
- the rejected-entity memory and attempt counters of documents not yet confirmed.

Neither is worth the cost. The first is a single document; the second simply starts over, which
is acceptable. Confirmed new entities are not lost, since they are created in Paperless-ngx on
confirmation. In exchange, the design avoids a new dependency, a checkpoint file and its
lifecycle, a fresh-vs-resume branch in `run_interactive`, handling a completed thread under the
fixed `thread_id`, and detecting input-folder drift between runs (each run simply lists the
folder again).

Remaining window: if the process dies after `upload_document` succeeds but before the file is
moved or deleted, a restart uploads that one document again. This is accepted as a known
limitation.

## Files touched

- `strategies/iterative/graph.py` — route `review_current_proposal` → `persist_new_entities` →
  `persist_file_proposals` → conditional edge to `classify_current_document` or `END`; remove
  the old `persist_file_proposals` → `END` edge and the conditional edge off
  `review_current_proposal`.
- `core/nodes/resultpersistence.py` — move to `processed/` (with collision suffix) when not
  deleting.
- `core/nodes/entities.py` — `load_documents` skips non-files.
- `pyproject.toml` — add `[project.optional-dependencies] dev = ["pytest>=9.1.1"]`, identical to
  `perf/classification-prefix-caching` so the branches merge cleanly.
- `README.md` — document that confirmed documents are uploaded immediately and that, unless
  `DELETE_INPUT_FILES_AFTER_UPLOAD=true`, uploaded files are moved to `INPUT_FOLDER/processed/`.

## Testing

Mock mode (`ACCOUNT`/`PASSWORD` containing `"mock"`, entities from `test/paperless-instance-mock`)
with pytest, under `test/paperlesslabelagent/`:

- `persist_file_proposals` against a temp input folder:
  - a confirmed file is moved to `processed/` when the delete flag is unset;
  - a confirmed file is deleted when `DELETE_INPUT_FILES_AFTER_UPLOAD=true`;
  - a name clash in `processed/` produces a suffixed name and leaves the existing file intact;
  - unconfirmed files stay in the input folder.
- `load_documents` ignores a `processed/` subfolder.
- End-to-end mock run of the iterative graph: confirm some documents, stop the run, and verify
  those files are already (simulated-)uploaded and moved; then start a fresh run on the same
  folder and verify it only processes the remaining files. LLM classification and the review
  prompts are stubbed so the test is deterministic.

**Do not use the real configured input folder for this.** The local `.env` sets
`DELETE_INPUT_FILES_AFTER_UPLOAD=true`, so a bug during verification could delete real source
files. Use a throwaway folder (pytest `tmp_path`).
