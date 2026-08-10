"""Iterative execution strategy.

Unlike the sequential strategy (classify every document, then review them all), this
strategy processes one document fully - classify, review, accept/reject - before moving to
the next. As soon as a new entity is accepted, it's merged into the in-memory
`existingEntities` pool (see `core.entities.merge_confirmed_new_entities`) so later
documents can match against it directly instead of independently proposing a near-duplicate
(e.g. "Deutsche Bahn" vs. "DB AG").

See `graph.py` for the two-node loop (`classify_current_document` / `review_current_proposal`)
and `state.py` for this strategy's own state extensions (`IterativeAgentState`,
`IterativeFileProposal`).
"""
