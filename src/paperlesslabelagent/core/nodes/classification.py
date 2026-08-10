from typing import Any

from langchain_ollama import ChatOllama

from paperlesslabelagent.core.config import MODEL, ENTITY_LANGUAGE
from paperlesslabelagent.core.schemas import MatchModel, NewEntityProposalModel, build_new_entities_model
from paperlesslabelagent.core.state import FileProposal

# Documents whose extracted text exceeds this many characters are summarized by the
# summarize_document_text subagent before being handed to the matcher, to keep the
# classification prompt within the model's context window.
MAX_DOCUMENT_CHARS = 30000


MATCH_SYSTEM_PROMPT = """You are an assistant that matches documents to existing tags, correspondents and document types for Paperless-ngx.
For the given document text, decide which existing tags, correspondent and document type fit.

Rules:
- Carefully check the document text against every existing entity in the provided lists before deciding nothing fits.
- Only choose an existing tag/correspondent/document type if you are reasonably confident it fits (confidence >= 0.6). Reference it by its exact id from the provided list.
- Never invent an id or name that is not present in the provided list of existing entities.
- A document can have zero, one or several tags.
- A document has at most one correspondent and at most one document type.
- If the prompt lists entities the user already rejected for this document, do not propose them again.
"""

NEW_ENTITY_SYSTEM_PROMPT = f"""You are an assistant that proposes new tags, correspondents and/or document types for Paperless-ngx, for a document where no existing entity was a good fit.

Rules:
- Only propose tags, correspondents and/or document types for the categories listed below - every category you're asked about had no confident existing match.
- Do not propose a new tag, correspondent or document type if a suitable existing one already exists in the provided reference list; in that case leave the field empty (null, or an empty list for tags).
- For new entities only use names from the languages {ENTITY_LANGUAGE}. Only propose terms in other languages, when REALLY no applicable term in {ENTITY_LANGUAGE} is available.
- Avoid additions to the entity name like "(or similar)" or an explanation.
- If the prompt lists newly-proposed entities the user already rejected for this document, do not propose them again.
"""

SUMMARIZER_SYSTEM_PROMPT = """You are an assistant that provides the parts of the given document text to support the extraction of labels, document_types and correspondents for Paperless-ngx.

Rules:
- Provide the text of the document, focusing on information that helps identify the appropriate labels, document_types and correspondents. Avoid including any irrelevant details or personal opinions.
- Only provide the text that is relevant for classification
- Do not addy any additional commentary or explanation.
- Do not yourself provide the labels, document_types or correspondents. Only provide the text that is relevant for classification."""

matcher = ChatOllama(model=MODEL, num_ctx=32768, temperature=0.3).with_structured_output(MatchModel)
new_entity_proposer = ChatOllama(model=MODEL, num_ctx=32768, temperature=0.3)
summarizer = ChatOllama(model=MODEL, num_ctx=32768, temperature=0.3)



def format_entity(entities: dict[str, Any]) -> str:
    items = entities.get("results", [])
    if not items:
        return "(none yet)"
    return "\n".join(f'- id={item["id"]}, name="{item["name"]}"' for item in items)


def format_rejected_existing_entities(items: list[dict[str, Any]]) -> str:
    """Formats a list of ExistingMatch entries the user already rejected for a document. Returns "" (nothing rendered) if there's nothing to report."""
    if not items:
        return ""
    lines = "\n".join(f'- id={item["id"]}, name="{item["name"]}"' for item in items)
    return f"\nAlready rejected by the user for this document - do not propose these again:\n{lines}\n"


def format_rejected_new_entities(items: list[dict[str, Any]]) -> str:
    """Formats a list of NewEntityProposal entries the user already rejected for a document. Returns "" (nothing rendered) if there's nothing to report."""
    if not items:
        return ""
    lines = "\n".join(f'- "{item["name"]}" - {item["description"]}' for item in items)
    return f"\nAlready rejected by the user for this document - do not propose these again:\n{lines}\n"


def build_user_prompt(
    filename: str,
    text: str,
    existing_entities: dict[str, Any],
    *,
    rejected_existing_tags: list[dict[str, Any]] | None = None,
    rejected_existing_correspondent: dict[str, Any] | None = None,
    rejected_existing_document_type: dict[str, Any] | None = None,
    is_summary: bool = False,
) -> str:
    tags = format_entity(existing_entities.get("tags", {}))
    correspondents = format_entity(existing_entities.get("correspondents", {}))
    document_types = format_entity(existing_entities.get("document_types", {}))

    rejected_tags_section = format_rejected_existing_entities(rejected_existing_tags or [])
    rejected_correspondent_section = format_rejected_existing_entities([rejected_existing_correspondent] if rejected_existing_correspondent else [])
    rejected_document_type_section = format_rejected_existing_entities([rejected_existing_document_type] if rejected_existing_document_type else [])

    text_label = "Document text (summarized, the original was too long to include in full)" if is_summary else "Document text"

    return f"""Document: {filename}

Existing tags:
{tags}
{rejected_tags_section}
Existing correspondents:
{correspondents}
{rejected_correspondent_section}
Existing document types:
{document_types}
{rejected_document_type_section}
{text_label}:
{text}
"""


def build_new_entities_user_prompt(
    filename: str,
    text: str,
    existing_entities: dict[str, Any],
    *,
    include_tags: bool,
    include_correspondent: bool,
    include_document_type: bool,
    rejected_new_tags: list[dict[str, Any]] | None = None,
    rejected_new_correspondent: dict[str, Any] | None = None,
    rejected_new_document_type: dict[str, Any] | None = None,
    is_summary: bool = False,
) -> str:
    """Builds the prompt for the new-entity-proposal step, listing only the existing-entity
    reference lists for the categories that had no match (the ones actually being asked about)."""
    sections = [f"Document: {filename}\n"]
    requested = []

    if include_tags:
        sections.append(f"Existing tags:\n{format_entity(existing_entities.get('tags', {}))}\n")
        rejected_section = format_rejected_new_entities(rejected_new_tags or [])
        if rejected_section:
            sections.append(rejected_section)
        requested.append("new tags")
    if include_correspondent:
        sections.append(f"Existing correspondents:\n{format_entity(existing_entities.get('correspondents', {}))}\n")
        rejected_section = format_rejected_new_entities([rejected_new_correspondent] if rejected_new_correspondent else [])
        if rejected_section:
            sections.append(rejected_section)
        requested.append("a new correspondent")
    if include_document_type:
        sections.append(f"Existing document types:\n{format_entity(existing_entities.get('document_types', {}))}\n")
        rejected_section = format_rejected_new_entities([rejected_new_document_type] if rejected_new_document_type else [])
        if rejected_section:
            sections.append(rejected_section)
        requested.append("a new document type")

    sections.append(f"No existing match was found for: {', '.join(requested)}. Propose new entities for these categories only.\n")

    text_label = "Document text (summarized, the original was too long to include in full)" if is_summary else "Document text"
    sections.append(f"{text_label}:\n{text}")

    return "\n".join(sections)




def summarize_document_text(text: str) -> str:
    """
    Summarizes the already-extracted text of a document, keeping only the information relevant
    to choosing tags, a correspondent and a document type for Paperless-ngx.
    """
    result = summarizer.invoke(
        [
            {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
    )
    return result.content


def classify_document(
    filename: str,
    text: str,
    existing_entities: dict[str, Any],
    *,
    rejected_existing_tags: list[dict[str, Any]],
    rejected_existing_correspondent: dict[str, Any] | None,
    rejected_existing_document_type: dict[str, Any] | None,
    rejected_new_tags: list[dict[str, Any]],
    rejected_new_correspondent: dict[str, Any] | None,
    rejected_new_document_type: dict[str, Any] | None,
) -> FileProposal:
    """Classifies a single document: matches it against existing entities, and for any category with no confident match, proposes a new entity instead. Returns a FileProposal, without touching any broader `proposals` collection itself.
    """
    is_summary = len(text) > MAX_DOCUMENT_CHARS
    if is_summary:
        print(f'Document "{filename}" is {len(text)} characters long (limit {MAX_DOCUMENT_CHARS}) — summarizing before classification.')
        text = summarize_document_text(text)
        print(f'Summarized text for "{filename}":\n{text}')

    match_prompt = build_user_prompt(
        filename,
        text,
        existing_entities,
        rejected_existing_tags=rejected_existing_tags,
        rejected_existing_correspondent=rejected_existing_correspondent,
        rejected_existing_document_type=rejected_existing_document_type,
        is_summary=False,
    )
    match_result: MatchModel = matcher.invoke(
        [
            {"role": "system", "content": MATCH_SYSTEM_PROMPT},
            {"role": "user", "content": match_prompt},
        ]
    )

    needs_tags = not match_result.tags
    needs_correspondent = match_result.correspondent is None
    needs_document_type = match_result.document_type is None

    new_tags: list[NewEntityProposalModel] = []
    new_correspondent: NewEntityProposalModel | None = None
    new_document_type: NewEntityProposalModel | None = None

    if needs_tags or needs_correspondent or needs_document_type:
        new_entities_model = build_new_entities_model(
            include_tags=needs_tags,
            include_correspondent=needs_correspondent,
            include_document_type=needs_document_type,
        )
        new_entities_prompt = build_new_entities_user_prompt(
            filename,
            text,
            existing_entities,
            include_tags=needs_tags,
            include_correspondent=needs_correspondent,
            include_document_type=needs_document_type,
            rejected_new_tags=rejected_new_tags,
            rejected_new_correspondent=rejected_new_correspondent,
            rejected_new_document_type=rejected_new_document_type,
            is_summary=False,
        )
        new_result = new_entity_proposer.with_structured_output(new_entities_model).invoke(
            [
                {"role": "system", "content": NEW_ENTITY_SYSTEM_PROMPT},
                {"role": "user", "content": new_entities_prompt},
            ]
        )
        new_tags = getattr(new_result, "new_tags", [])
        new_correspondent = getattr(new_result, "new_correspondent", None)
        new_document_type = getattr(new_result, "new_document_type", None)

    return {
        "filename": filename,
        "proposed_existing_tags": [tag.model_dump() for tag in match_result.tags],
        "proposed_existing_correspondent": match_result.correspondent.model_dump() if match_result.correspondent else None,
        "proposed_existing_document_type": match_result.document_type.model_dump() if match_result.document_type else None,
        "proposed_new_tags": [proposal.model_dump() for proposal in new_tags] or None,
        "proposed_new_correspondent": new_correspondent.model_dump() if new_correspondent else None,
        "proposed_new_document_type": new_document_type.model_dump() if new_document_type else None,
        "confirmed": False,
        "needs_retry": False,
        "rejected_existing_tags": rejected_existing_tags,
        "rejected_existing_correspondent": rejected_existing_correspondent,
        "rejected_existing_document_type": rejected_existing_document_type,
        "rejected_new_tags": rejected_new_tags,
        "rejected_new_correspondent": rejected_new_correspondent,
        "rejected_new_document_type": rejected_new_document_type,
    }
