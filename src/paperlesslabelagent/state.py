from typing import Any, Literal, TypedDict

EntityType = Literal["tag", "correspondent", "document_type"]


class ExistingMatch(TypedDict):
    """A proposed match against an entity that already exists in the set of tags, correspondents and document types"""

    id: int
    name: str
    confidence: float
    reasoning: str


class NewEntityProposal(TypedDict):
    """A proposed new tag, correspondent or document type"""

    entity_type: EntityType
    name: str
    description: str
    reasoning: str


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

    iteration: int
    upload_results: dict[str, UploadResult]
