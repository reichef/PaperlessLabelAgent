from typing import Any, Literal, NotRequired, TypedDict

EntityType = Literal["tag", "correspondent", "document_type"]


class ExistingMatch(TypedDict):
    """A proposed match against an entity that already exists in the set of tags, correspondents and document types"""

    name: str
    confidence: float
    reasoning: str
    # Only present once the match has been verified against existing_entities (see
    # check_and_correct_single_proposal) - the LLM itself never produces this, it only knows names.
    id: NotRequired[int]


class NewEntityProposal(TypedDict):
    """A proposed new tag, correspondent or document type"""

    entity_type: EntityType
    name: str
    description: str
    reasoning: str
    # Only present once the entity has been folded into existing_entities with a placeholder id
    # (see merge_confirmed_new_entities), later replaced by the real Paperless-ngx id on persist.
    id: NotRequired[int]


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
