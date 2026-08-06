from typing import Any

from pydantic import BaseModel, Field, create_model

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


class MatchModel(BaseModel):
    """Structured output schema for the match-only step: which existing tags, correspondent
    and document type (if any) fit the document.

    Fields are required (not defaulted) rather than optional: with Ollama's JSON-schema-
    constrained decoding, a schema where every field is optional lets the model emit a
    trivially valid empty object and stop engaging with the task. Requiring the key to be
    present (its value may still be null or an empty list) forces the model to make an
    explicit decision for every field instead.
    """

    tags: list[ExistingMatchModel]
    correspondent: ExistingMatchModel | None
    document_type: ExistingMatchModel | None


def build_new_entities_model(*, include_tags: bool, include_correspondent: bool, include_document_type: bool) -> type[BaseModel]:
    """Builds a structured-output schema for the new-entity-proposal step, containing only the
    fields for categories that had no match in the preceding MatchModel step.

    Only including the relevant fields means the model isn't asked about categories that are
    already resolved, and (per MatchModel's docstring) keeping them required rather than
    defaulted keeps the model from skipping the ones it is asked about.
    """
    fields: dict[str, Any] = {}
    if include_tags:
        fields["new_tags"] = (list[NewEntityProposalModel], ...)
    if include_correspondent:
        fields["new_correspondent"] = (NewEntityProposalModel | None, ...)
    if include_document_type:
        fields["new_document_type"] = (NewEntityProposalModel | None, ...)
    return create_model("NewEntitiesModel", **fields)
