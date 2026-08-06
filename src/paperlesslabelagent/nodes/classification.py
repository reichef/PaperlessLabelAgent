from typing import Any

from langchain_ollama import ChatOllama

from paperlesslabelagent.config import MODEL
from paperlesslabelagent.schemas import MatchModel, NewEntityProposalModel, build_new_entities_model
from paperlesslabelagent.state import AgentState, FileProposal
from paperlesslabelagent.tools.pdftools import summarize_document_text

# Documents whose extracted text exceeds this many characters are summarized by the
# summarize_document_text subagent before being handed to the matcher, to keep the
# classification prompt within the model's context window.
MAX_DOCUMENT_CHARS = 10000


MATCH_SYSTEM_PROMPT = """You are an assistant that matches documents to existing tags, correspondents and document types for Paperless-ngx.
For the given document text, decide which existing tags, correspondent and document type fit.

Rules:
- Carefully check the document text against every existing entity in the provided lists before deciding nothing fits.
- Only choose an existing tag/correspondent/document type if you are reasonably confident it fits (confidence >= 0.6). Reference it by its exact id from the provided list.
- Never invent an id or name that is not present in the provided list of existing entities.
- A document can have zero, one or several tags.
- A document has at most one correspondent and at most one document type.
"""

NEW_ENTITY_SYSTEM_PROMPT = """You are an assistant that proposes new tags, correspondents and/or document types for Paperless-ngx, for a document where no existing entity was a good fit.

Rules:
- Only propose tags, correspondents and/or document types for the categories listed below - every category you're asked about had no confident existing match.
- Do not propose a new tag, correspondent or document type if a suitable existing one already exists in the provided reference list; in that case leave the field empty (null, or an empty list for tags).
- Avoid additions to the name like "(or similar)" or an explanation.
"""

matcher = ChatOllama(model=MODEL, num_ctx=32768, temperature=0.3).with_structured_output(MatchModel)
proposer = ChatOllama(model=MODEL, num_ctx=32768, temperature=0.3)


def format_entity(entities: dict[str, Any]) -> str:
    items = entities.get("results", [])
    if not items:
        return "(none yet)"
    return "\n".join(f'- id={item["id"]}, name="{item["name"]}"' for item in items)


def build_user_prompt(filename: str, text: str, existing_entities: dict[str, Any], is_summary: bool = False) -> str:
    tags = format_entity(existing_entities.get("labels", {}))
    correspondents = format_entity(existing_entities.get("correspondents", {}))
    document_types = format_entity(existing_entities.get("document_types", {}))

    text_label = "Document text (summarized, the original was too long to include in full)" if is_summary else "Document text"

    return f"""Document: {filename}

Existing tags:
{tags}

Existing correspondents:
{correspondents}

Existing document types:
{document_types}

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
    is_summary: bool = False,
) -> str:
    """Builds the prompt for the new-entity-proposal step, listing only the existing-entity
    reference lists for the categories that had no match (the ones actually being asked about)."""
    sections = [f"Document: {filename}\n"]
    requested = []

    if include_tags:
        sections.append(f"Existing tags:\n{format_entity(existing_entities.get('labels', {}))}\n")
        requested.append("new tags")
    if include_correspondent:
        sections.append(f"Existing correspondents:\n{format_entity(existing_entities.get('correspondents', {}))}\n")
        requested.append("a new correspondent")
    if include_document_type:
        sections.append(f"Existing document types:\n{format_entity(existing_entities.get('document_types', {}))}\n")
        requested.append("a new document type")

    sections.append(f"No existing match was found for: {', '.join(requested)}. Propose new entities for these categories only.\n")

    text_label = "Document text (summarized, the original was too long to include in full)" if is_summary else "Document text"
    sections.append(f"{text_label}:\n{text}")

    return "\n".join(sections)


def propose_for_document(state: AgentState) -> dict[str, Any]:
    """Step 3: For each not-yet-confirmed file, propose matching (or new) tags, correspondent and document type."""
    existing_entities = state["existingEntities"]
    proposals: dict[str, FileProposal] = dict(state.get("proposals", {}))

    for filename, text in state["file_texts"].items():
        if proposals.get(filename, {}).get("confirmed"):
            continue

        # Currently, LLM Summary is disabled as it made the classification worse. We keep the code here for future improvement.

        is_summary = len(text) > MAX_DOCUMENT_CHARS
        if is_summary:
            print(f'Document "{filename}" is {len(text)} characters long (limit {MAX_DOCUMENT_CHARS}) — summarizing before classification.')
            text = summarize_document_text(text)
            print(f'Summarized text for "{filename}":\n{text}')

        match_prompt = build_user_prompt(filename, text, existing_entities, is_summary=False)
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
                is_summary=False,
            )
            new_result = proposer.with_structured_output(new_entities_model).invoke(
                [
                    {"role": "system", "content": NEW_ENTITY_SYSTEM_PROMPT},
                    {"role": "user", "content": new_entities_prompt},
                ]
            )
            new_tags = getattr(new_result, "new_tags", [])
            new_correspondent = getattr(new_result, "new_correspondent", None)
            new_document_type = getattr(new_result, "new_document_type", None)

        proposals[filename] = {
            "filename": filename,
            "tags": [tag.model_dump() for tag in match_result.tags],
            "correspondent": match_result.correspondent.model_dump() if match_result.correspondent else None,
            "document_type": match_result.document_type.model_dump() if match_result.document_type else None,
            "new_tags": [proposal.model_dump() for proposal in new_tags] or None,
            "new_correspondent": new_correspondent.model_dump() if new_correspondent else None,
            "new_document_type": new_document_type.model_dump() if new_document_type else None,
            "confirmed": False,
            "needs_retry": False,
        }

    return {"proposals": proposals}
