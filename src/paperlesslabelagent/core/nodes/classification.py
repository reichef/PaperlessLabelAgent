from typing import Any

from langchain_ollama import ChatOllama

from paperlesslabelagent.core.config import MODEL, ENTITY_LANGUAGE
from paperlesslabelagent.core.schemas import MatchModel, NewEntityProposalModel, build_new_entities_model
from paperlesslabelagent.core.state import ExistingMatch, FileProposal, NewEntityProposal

# Describes the length of document text that is allowed to be passed to a classification LLM without summarization.
# After this length, the text is summarized first, and the summary is passed to the LLM for classification instead. 
# This avoids drawbacks of passing too much context into the LLM at once.
MAX_DOCUMENT_CHARS = 30000


# Identical for the match call and the new-entity call, and always the very first content in
# both requests, so the two share a cacheable prefix on a single-instance Ollama server: the
# new-entity call's prefill can reuse the match call's KV-cache instead of reprocessing the
# document text and entity lists from scratch. Task-specific rules therefore live in the
# per-call instructions appended at the END of the user prompt (see MATCH_INSTRUCTIONS /
# NEW_ENTITY_INSTRUCTIONS below), not here.
SYSTEM_PROMPT = """You are an assistant that classifies documents for Paperless-ngx: matching them against existing tags, correspondents and document types, and - only when explicitly instructed at the end of the prompt - proposing new ones for categories with no confident existing match.

General rules:
- Never invent a name that is not present in the provided list of existing entities when matching.
- A document can have zero, one or several tags; at most one correspondent; at most one document type.
- Follow the task-specific instructions at the end of the prompt exactly; they define which of the two tasks above you are being asked to perform right now.
"""

MATCH_INSTRUCTIONS = """Task: match the document above against the existing tags, correspondent and document type listed above.

- Carefully check the document text against every existing entity in the provided lists before deciding nothing fits.
- Only choose an existing tag/correspondent/document type if you are reasonably confident it fits (confidence >= 0.6). Reference it by its exact name from the provided list.
- If entities the user already rejected are listed above, do not propose them again.
"""

NEW_ENTITY_INSTRUCTIONS = f"""Task: no confident existing match was found for: {{requested}}. Propose new entities for these categories only.

- Do not propose a new tag, correspondent or document type if a suitable existing one already exists in the provided reference lists above; in that case leave the field empty (null, or an empty list for tags).
- For new entities only use names from the languages {ENTITY_LANGUAGE}. Only propose terms in other languages, when no applicable term in {ENTITY_LANGUAGE} is available.
- Avoid additions to the entity name like "(or similar)" or an explanation.
- If newly-proposed entities the user already rejected are listed above, do not propose them again.
"""

SUMMARIZER_SYSTEM_PROMPT = """You are an assistant that provides the parts of the given document text to support the extraction of tags, document_types and correspondents for Paperless-ngx.

Rules:
- Provide the text of the document, focusing on information that helps identify the appropriate tags, document_types and correspondents. Avoid including any irrelevant details or personal opinions.
- Only provide the text that is relevant for classification
- Do not addy any additional commentary or explanation.
- Do not yourself provide the tags, document_types or correspondents. Only provide the text that is relevant for classification."""

matcher = ChatOllama(model=MODEL, num_ctx=32768, temperature=0.3).with_structured_output(MatchModel)
new_entity_proposer = ChatOllama(model=MODEL, num_ctx=32768, temperature=0.3)
summarizer = ChatOllama(model=MODEL, num_ctx=32768, temperature=0.3)



def format_entity(entities: dict[str, Any]) -> str:
    items = entities.get("results", [])
    if not items:
        return "(none yet)"
    return "\n".join(f'- {item["name"]}' for item in items)


def format_rejected_existing_entities(items: list[ExistingMatch]) -> str:
    """Formats a list of ExistingMatch entries the user already rejected for a document. Returns "" (nothing rendered) if there's nothing to report."""
    if not items:
        return ""
    lines = "\n".join(f'- "{item.name}"' for item in items)
    return f"\nAlready rejected by the user for this document - do not propose these again:\n{lines}\n"


def format_rejected_new_entities(items: list[NewEntityProposal]) -> str:
    """Formats a list of NewEntityProposal entries the user already rejected for a document. Returns "" (nothing rendered) if there's nothing to report."""
    if not items:
        return ""
    lines = "\n".join(f'- "{item.name}" - {item.description}' for item in items)
    return f"\nAlready rejected by the user for this document - do not propose these again:\n{lines}\n"


def build_shared_context_block(
    filename: str,
    text: str,
    existing_entities: dict[str, Any],
    *,
    is_summary: bool = False,
) -> str:
    """Builds the part of the user prompt that is identical between the match call and the
    new-entity call - filename, all three existing-entity lists and the document text - with
    nothing call-specific mixed in, so it can be reused byte-for-byte as the leading prefix of
    both prompts (see SYSTEM_PROMPT for why that matters)."""
    tags = format_entity(existing_entities.get("tags", {}))
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


def build_user_prompt(
    filename: str,
    text: str,
    existing_entities: dict[str, Any],
    *,
    rejected_existing_tags: list[ExistingMatch] | None = None,
    rejected_existing_correspondent: ExistingMatch | None = None,
    rejected_existing_document_type: ExistingMatch | None = None,
    is_summary: bool = False,
) -> str:
    shared_block = build_shared_context_block(filename, text, existing_entities, is_summary=is_summary)

    rejected_tags_section = format_rejected_existing_entities(rejected_existing_tags or [])
    rejected_correspondent_section = format_rejected_existing_entities([rejected_existing_correspondent] if rejected_existing_correspondent else [])
    rejected_document_type_section = format_rejected_existing_entities([rejected_existing_document_type] if rejected_existing_document_type else [])

    return f"{shared_block}\n{rejected_tags_section}{rejected_correspondent_section}{rejected_document_type_section}\n{MATCH_INSTRUCTIONS}"


def build_new_entities_user_prompt(
    filename: str,
    text: str,
    existing_entities: dict[str, Any],
    *,
    include_tags: bool,
    include_correspondent: bool,
    include_document_type: bool,
    rejected_new_tags: list[NewEntityProposal] | None = None,
    rejected_new_correspondent: NewEntityProposal | None = None,
    rejected_new_document_type: NewEntityProposal | None = None,
    is_summary: bool = False,
) -> str:
    """Builds the prompt for the new-entity-proposal step. Shares its leading context block
    byte-for-byte with build_user_prompt (all three existing-entity lists, even for categories
    not being asked about here) so the two calls can share a cacheable prompt prefix; the
    trailing task instructions scope the model to only the requested categories."""
    shared_block = build_shared_context_block(filename, text, existing_entities, is_summary=is_summary)

    requested = []
    rejected_sections = ""
    if include_tags:
        rejected_sections += format_rejected_new_entities(rejected_new_tags or [])
        requested.append("new tags")
    if include_correspondent:
        rejected_sections += format_rejected_new_entities([rejected_new_correspondent] if rejected_new_correspondent else [])
        requested.append("a new correspondent")
    if include_document_type:
        rejected_sections += format_rejected_new_entities([rejected_new_document_type] if rejected_new_document_type else [])
        requested.append("a new document type")

    instructions = NEW_ENTITY_INSTRUCTIONS.format(requested=", ".join(requested))

    return f"{shared_block}\n{rejected_sections}\n{instructions}"




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
    rejected_existing_tags: list[ExistingMatch],
    rejected_existing_correspondent: ExistingMatch | None,
    rejected_existing_document_type: ExistingMatch | None,
    rejected_new_tags: list[NewEntityProposal],
    rejected_new_correspondent: NewEntityProposal | None,
    rejected_new_document_type: NewEntityProposal | None,
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": match_prompt},
        ]
    )


    # If no entities have been proposed correctly, run the request for proposals of new entities.
    needs_new_tags = not match_result.tags
    needs_new_correspondent = match_result.correspondent is None
    needs_new_document_type = match_result.document_type is None

    new_tags: list[NewEntityProposalModel] = []
    new_correspondent: NewEntityProposalModel | None = None
    new_document_type: NewEntityProposalModel | None = None

    if needs_new_tags or needs_new_correspondent or needs_new_document_type:
        new_entities_model = build_new_entities_model(
            include_tags=needs_new_tags,
            include_correspondent=needs_new_correspondent,
            include_document_type=needs_new_document_type,
        )
        new_entities_prompt = build_new_entities_user_prompt(
            filename,
            text,
            existing_entities,
            include_tags=needs_new_tags,
            include_correspondent=needs_new_correspondent,
            include_document_type=needs_new_document_type,
            rejected_new_tags=rejected_new_tags,
            rejected_new_correspondent=rejected_new_correspondent,
            rejected_new_document_type=rejected_new_document_type,
            is_summary=False,
        )
        new_result = new_entity_proposer.with_structured_output(new_entities_model).invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": new_entities_prompt},
            ]
        )
        new_tags = getattr(new_result, "new_tags", [])
        new_correspondent = getattr(new_result, "new_correspondent", None)
        new_document_type = getattr(new_result, "new_document_type", None)

    return {
        "filename": filename,
        "proposed_existing_tags": [ExistingMatch(name=tag.name, confidence=tag.confidence, reasoning=tag.reasoning) for tag in match_result.tags],
        "proposed_existing_correspondent": (
            ExistingMatch(name=match_result.correspondent.name, confidence=match_result.correspondent.confidence, reasoning=match_result.correspondent.reasoning)
            if match_result.correspondent else None
        ),
        "proposed_existing_document_type": (
            ExistingMatch(name=match_result.document_type.name, confidence=match_result.document_type.confidence, reasoning=match_result.document_type.reasoning)
            if match_result.document_type else None
        ),
        "proposed_new_tags": [
            NewEntityProposal(entity_type=proposal.entity_type, name=proposal.name, description=proposal.description, reasoning=proposal.reasoning)
            for proposal in new_tags
        ] or None,
        "proposed_new_correspondent": (
            NewEntityProposal(entity_type=new_correspondent.entity_type, name=new_correspondent.name, description=new_correspondent.description, reasoning=new_correspondent.reasoning)
            if new_correspondent else None
        ),
        "proposed_new_document_type": (
            NewEntityProposal(entity_type=new_document_type.entity_type, name=new_document_type.name, description=new_document_type.description, reasoning=new_document_type.reasoning)
            if new_document_type else None
        ),
        "confirmed": False,
        "needs_retry": False,
        "rejected_existing_tags": rejected_existing_tags,
        "rejected_existing_correspondent": rejected_existing_correspondent,
        "rejected_existing_document_type": rejected_existing_document_type,
        "rejected_new_tags": rejected_new_tags,
        "rejected_new_correspondent": rejected_new_correspondent,
        "rejected_new_document_type": rejected_new_document_type,
    }
