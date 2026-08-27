# Resuming an interrupted iterative run without losing or redoing confirmed work

## Problem

The iterative strategy (`strategies/iterative/graph.py`) only pushes results to Paperless-ngx
once, after the entire document loop finishes: `review_current_proposal` loops back to
`classify_current_document` for as long as `select_next_document` finds work, and only routes
to `persist_new_entities` → `persist_file_proposals` once nothing is left. If the agent is
aborted (crash, killed process, closed terminal) before that point, every classification
confirmed so far is lost — nothing was ever written to Paperless-ngx.

Separately, the graph's checkpointer is `InMemorySaver`, which keeps checkpoint state purely in
the Python process's memory. Even though this graph already checkpoints constantly (it
interrupts on essentially every document, for hallucination checks and review), none of that
survives a process restart. A fresh invocation starts `state["proposals"]` empty, so it would
redo LLM classification and human review for documents a previous, aborted run had already
confirmed.

These are two distinct failure modes bundled under "aborting the agent loses work":

- **(A)** the agent can't resume its own progress after a restart — already-confirmed documents
  get reclassified and re-reviewed from scratch.
- **(B)** confirmed classifications don't reach Paperless-ngx until the entire batch finishes —
  an abort loses real, human-reviewed work, not just recomputable agent state.

They're complementary, not redundant: (B) alone prevents *data loss* but not *wasted
recomputation or duplicate uploads* on restart (see the failure scenario in "Design" below); (A)
alone prevents wasted recomputation but doesn't get already-reviewed work into Paperless-ngx any
earlier than today.

## Goal

- A confirmed document's classification is applied to Paperless-ngx immediately, not batched to
  the end of a run.
- Restarting the agent after a crash/abort resumes exactly where the previous run left off —
  already-confirmed documents are neither reclassified/re-reviewed nor re-uploaded.
- The agent detects when the input folder's contents have drifted from what the interrupted run
  started with (files added or removed while the process was down) and reports this explicitly
  rather than silently ignoring it or crashing on a missing file.

## Non-goals

- The sequential strategy (`strategies/sequential`). It classifies every document before
  reviewing any, which is a fundamentally different loss profile, and wasn't the strategy this
  design was scoped to.
- Detecting content changes to a file that stays present under the same filename (e.g. someone
  replaces `invoice.pdf` with different content while the agent is paused). Only
  presence/absence in the input folder is checked, not content.
- Automatically merging newly-added files into an in-flight resumed run. They're reported and
  left for a separate, later run rather than spliced into already-checkpointed state.
- Concurrent or multi-instance execution against the same run. This design assumes a single
  agent process at a time, consistent with today's single fixed
  `thread_id="paperless-label-run"` (`agent.py`).

## Design

### (B) Per-document eager persistence

Move `persist_new_entities` → `persist_file_proposals` inside the per-document loop, running
immediately after each document is confirmed, instead of once after the whole loop exits.

This is safe per-document, not just per-batch, because `merge_confirmed_new_entities` (already
called from `review_current_proposal` on confirmation) folds a confirmed document's new entities
into `existingEntities` — with placeholder ids — before that node returns. So by the time
`persist_new_entities` runs for a given document, it only ever needs to resolve *that*
document's own placeholders; it doesn't need to reconcile duplicates proposed independently by
several documents, because a later document's classification call already matches against the
earlier one's (not-yet-real) entity as an existing entity, not a fresh proposal. The batch-level
dedupe-by-name logic already in `persist_new_entities` stays as a safety net, but stops being
load-bearing.

Trade-off: this removes the current "nothing touches live Paperless-ngx until the whole batch is
reviewed" property. A confirmed document is live immediately, with no in-agent undo — fixing a
mistaken confirmation afterward means editing it directly in Paperless-ngx. This is an
intentional trade in favor of durability.

Both persist nodes are called unconditionally after every `review_current_proposal`, including
iterations that route back for a retry rather than a confirmation — this stays safe without extra
routing logic because `persist_file_proposals` already skips any proposal where
`confirmed` is falsy, and `persist_new_entities` only ever has placeholder ids to resolve for
entities `merge_confirmed_new_entities` folded in, which only happens on confirmation. A retry
iteration's persist calls are therefore genuine no-ops, not something the graph needs to
distinguish.

### (A) Durable checkpointing

Swap `InMemorySaver` for `SqliteSaver` (`langgraph-checkpoint-sqlite`), pointed at a checkpoint
file on ordinary persistent storage (not a temp directory — a file there would not survive a
reboot, defeating the point). LangGraph checkpoints after every superstep, and this graph already
interrupts on essentially every document, so with a durable backend a crash loses at most the one
document that was mid-classification or mid-review when it happened.

`agent.py` already uses a fixed `thread_id="paperless-label-run"`, which is exactly the "same
run" identity durable checkpointing needs across process restarts.

`run_interactive` (`core/driver.py`) currently always starts with
`graph.invoke(initial_state, config=config)` — correct for a fresh run, but wrong for resuming
one paused at an interrupt: re-injecting `initial_state` into a thread that already has state
would not correctly continue from the pause point. Verified against the installed LangGraph
version (1.2.10): `graph.get_state(config)` returns a `StateSnapshot` with `.next` (empty once
the run has completed or if there's no prior checkpoint; non-empty while paused) and
`.interrupts` (a tuple of pending `Interrupt` objects, each carrying the same `.value` payload
already produced by `interrupt({...})` inside the graph — the same shape `run_interactive`
already reads via `result["__interrupt__"][0].value` during a live run). `run_interactive`
should branch on this at startup:

- `.next` empty → today's path: `graph.invoke(initial_state, config=config)`.
- `.next` non-empty → a prior run was interrupted mid-way. Skip the fresh invoke. Run the
  folder-consistency check below, then resume by feeding `.interrupts[0].value` into the
  existing `while "__interrupt__" in result` loop starting with `Command(resume=answer)`.

### Folder-consistency check on resume

Runs only on the resume path above, before continuing the graph. Compares the checkpointed
`state["file_texts"].keys()` (the original snapshot from when `load_documents` ran, at the start
of the interrupted run — available via the `StateSnapshot.values` already fetched above) against
a fresh `os.listdir(input_folder)` taken now, at resume time:

- **Missing, but the file's proposal is already confirmed** — expected. This is exactly what a
  successful eager-persist run looks like when `DELETE_INPUT_FILES_AFTER_UPLOAD=true`. No
  warning.
- **Missing, but still pending** (not confirmed) — the source file is gone. Drop the filename
  from the run's remaining work (`file_texts`/`proposals`) rather than letting
  `classify_current_document` or the upload step fail on a missing path.
- **New, not in the original snapshot** — report it, but do not pull it into this run (the
  `load_documents` node already ran once and won't run again on resume). It's left for a
  separate, later run.

Report both buckets and require an explicit y/n confirmation before resuming, consistent with
the existing `ask_yes_no`-based review/hallucination prompts in `core/nodes/review.py`, e.g.:

```
2 pending file(s) no longer in the input folder, skipping: X, Y.
3 new file(s) found, not included in this resumed run: A, B, C.
Continue resuming? [y/n]
```

This lives in `core/driver.py` as a plain CLI check, not a graph `interrupt()` — it's a
pre-flight decision about whether/how to resume the graph, made before the graph is invoked
again, not part of the graph's own state machine.

### Why (B) alone is not sufficient

Confirmed during brainstorming: suppose documents 1–5 are classified, reviewed, and (with (B)
alone) immediately persisted, then the process is killed before document 6. On restart, without
(A), `state["proposals"]` starts empty again. If `DELETE_INPUT_FILES_AFTER_UPLOAD=false` (the
documented default), files 1–5 are still in the input folder, so a fresh run would reclassify,
re-review, and — with eager persistence active — re-upload them, producing duplicate documents
in Paperless-ngx. (A) is what lets the agent know, before doing any work, which documents are
already confirmed.

## Files touched

- `strategies/iterative/graph.py` — reroute the edges so `review_current_proposal` always flows
  into `persist_new_entities` → `persist_file_proposals` first; `route_after_review`'s
  `select_next_document`-based branch (continue vs. end) moves to run *after* those two nodes
  instead of directly off `review_current_proposal`, routing to either
  `classify_current_document` (more documents left) or `END` (none left) — `persist_new_entities`
  no longer needs its own dedicated terminal edge into `persist_file_proposals` → `END`, since
  that pair is now inside the loop itself. Swap `InMemorySaver` for `SqliteSaver`, which needs a
  connection lifecycle (`SqliteSaver.from_conn_string(...)` is a context manager) in place of
  today's bare module-level `InMemorySaver()` instantiation.
- `core/driver.py` — `run_interactive` gains the `get_state`-based fresh-vs-resume branch and the
  folder-consistency check on the resume path.
- `agent.py` — owns/passes the sqlite checkpoint file path (new config, e.g.
  `CHECKPOINT_DB_PATH`); likely needs to open the sqlite connection here (or in a small setup
  helper) and pass it into graph compilation, given `SqliteSaver`'s context-manager lifecycle.
- `pyproject.toml` — new dependency `langgraph-checkpoint-sqlite`.
- `README.md` — document the new checkpoint-file config and the resume behavior.

## Open questions for the implementation plan

- Where the sqlite checkpoint file should live by default (repo-relative path vs. an env var),
  and whether/how it should be cleaned up once a run completes successfully (leaving it forever
  under a fixed `thread_id` means every future run "resumes" a completed thread rather than
  starting fresh, unless completion is detected and handled explicitly).
- Whether a `persist_new_entities`/`persist_file_proposals` failure mid-loop (e.g. a
  Paperless-ngx API error) should halt the whole run, or be retried like a classification
  failure (subject to `MAX_ATTEMPTS_PER_DOCUMENT`).

## Testing

Mirrors the mock-mode approach already used in this project (`test/paperless-instance-mock`,
`ACCOUNT`/`PASSWORD` containing `"mock"`), using the pytest setup introduced on
`perf/classification-prefix-caching`. Once implemented:

- A unit test that `run_interactive`, given a fake `StateSnapshot` with non-empty `.next`, takes
  the resume branch (extracts `.interrupts[0].value` and calls `Command(resume=...)`) rather
  than `graph.invoke(initial_state, ...)`.
- A unit test for the folder-consistency diff logic (missing-but-confirmed /
  missing-but-pending / new) against synthetic `file_texts` + `proposals` + a temp directory
  listing.
- An end-to-end mock-mode run of the iterative graph: run it against a `SqliteSaver` pointed at a
  temp file, stop partway through several documents (without calling further `graph.invoke`),
  then resume against the same `thread_id` and sqlite file, verifying already-confirmed
  documents are neither reclassified nor re-uploaded and the run completes correctly for the
  remainder.

**Do not use the real configured input folder for this.** If `DELETE_INPUT_FILES_AFTER_UPLOAD`
is enabled, a bug during verification could delete real source files. Use a throwaway input
folder outside the repo, as with prior verification work in this project.
