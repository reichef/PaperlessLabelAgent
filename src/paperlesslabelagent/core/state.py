from dataclasses import dataclass
from typing import Any, Literal, TypedDict

EntityType = Literal["tag", "correspondent", "document_type"]


@dataclass
class ExistingMatch:
    """A proposed match against an entity that already exists in the set of tags, correspondents and document types"""

    name: str
    confidence: float
    reasoning: str
    # Set by check_and_correct_single_proposal, resolved against existing_entities by name -
    # the LLM itself never produces this, it only knows names. This may still be a negative
    # placeholder id afterward: if the matched entity is itself a not-yet-persisted new entity
    # from an earlier proposal (folded into existing_entities by merge_confirmed_new_entities,
    # see core/nodes/entities.py), that's what's in the pool at match time. It's only
    # guaranteed to be a real, positive Paperless-ngx id once persist_new_entities'
    # propagation pass (core/nodes/resultpersistence.py) has run.
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


class FileProposal(TypedDict):
    """The agent's proposal for a single file's tags, correspondent and document type."""

    filename: str
    proposed_existing_tags: list[ExistingMatch]
    proposed_existing_correspondent: ExistingMatch | None
    proposed_existing_document_type: ExistingMatch | None
    proposed_new_tags: list[NewEntityProposal] | None
    proposed_new_correspondent: NewEntityProposal | None
    proposed_new_document_type: NewEntityProposal | None
    confirmed: bool
    needs_retry: bool
    rejected_existing_correspondent: ExistingMatch | None
    rejected_existing_document_type: ExistingMatch | None
    rejected_existing_tags: list[ExistingMatch]
    rejected_new_correspondent: NewEntityProposal | None
    rejected_new_document_type: NewEntityProposal | None
    rejected_new_tags: list[NewEntityProposal] | None


class UploadResult(TypedDict):
    """Result of submitting a file to Paperless-ngx for consumption."""

    task_id: str


class AgentState(TypedDict):
    input_folder: str

    existingEntities: dict[str, Any]
    file_texts: dict[str, str]

    proposals: dict[str, FileProposal]

    pending_new_entities: list[NewEntityProposal]
    confirmed_new_entities: list[NewEntityProposal]
    created_entity_ids: dict[str, int]

    upload_results: dict[str, UploadResult]
