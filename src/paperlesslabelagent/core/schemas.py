from typing import Any

from pydantic import BaseModel, Field, create_model

from paperlesslabelagent.core.state import EntityType


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


class MatchModel(BaseModel):
    """Structured output schema for the match-only step: which existing tags, correspondent
    and document type (if any) fit the document."""

    tags: list[ExistingMatchModel]
    correspondent: ExistingMatchModel | None
    document_type: ExistingMatchModel | None


def build_new_entities_model(*, include_tags: bool, include_correspondent: bool, include_document_type: bool) -> type[BaseModel]:
    """Builds a structured-output schema for the new-entity-proposal step, containing only the
    fields for categories that had no match in the preceding MatchModel step.
    """
    fields: dict[str, Any] = {}
    if include_tags:
        fields["new_tags"] = (list[NewEntityProposalModel], ...)
    if include_correspondent:
        fields["new_correspondent"] = (NewEntityProposalModel | None, ...)
    if include_document_type:
        fields["new_document_type"] = (NewEntityProposalModel | None, ...)
    return create_model("NewEntitiesModel", **fields)
