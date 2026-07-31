from pydantic import BaseModel, Field

from paperlesslabelagent.state import EntityType


class ExistingMatchModel(BaseModel):
    """A proposed match against an entity that already exists in Paperless-ngx."""

    id: int
    name: str
    confidence: float = Field(ge=0, le=1)
    reasoning: str


class NewEntityProposalModel(BaseModel):
    """A proposed new tag, correspondent or document type."""

    entity_type: EntityType
    name: str
    description: str
    reasoning: str


class FileProposalModel(BaseModel):
    """Structured output schema for the LLM's per-document matching decision."""

    tags: list[ExistingMatchModel] = Field(default_factory=list)
    correspondent: ExistingMatchModel | None = None
    document_type: ExistingMatchModel | None = None
    new_tags: list[NewEntityProposalModel] = Field(default_factory=list)
    new_correspondent: NewEntityProposalModel | None = None
    new_document_type: NewEntityProposalModel | None = None
