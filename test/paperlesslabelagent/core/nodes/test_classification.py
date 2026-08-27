from paperlesslabelagent.core.nodes import classification as classification_module
from paperlesslabelagent.core.nodes.classification import (
    build_new_entities_user_prompt,
    build_shared_context_block,
    build_user_prompt,
)


def _existing_entities() -> dict:
    return {
        "tags": {"results": [{"id": 1, "name": "Invoice"}]},
        "correspondents": {"results": [{"id": 2, "name": "Acme Corp"}]},
        "document_types": {"results": [{"id": 3, "name": "Letter"}]},
    }


def test_match_and_new_entity_prompts_share_identical_prefix():
    """The match prompt and the new-entity prompt must start with the exact same
    shared-context text so Ollama can reuse the first call's prefill for the second call."""
    filename = "sample.pdf"
    text = "Some extracted document text."
    entities = _existing_entities()

    shared_block = build_shared_context_block(filename, text, entities)
    match_prompt = build_user_prompt(filename, text, entities)
    new_entity_prompt = build_new_entities_user_prompt(
        filename,
        text,
        entities,
        include_tags=True,
        include_correspondent=True,
        include_document_type=True,
    )

    assert match_prompt.startswith(shared_block)
    assert new_entity_prompt.startswith(shared_block)


class _FakeMatcherClient:
    def __init__(self, result):
        self._result = result
        self.captured_messages = None

    def invoke(self, messages):
        self.captured_messages = messages
        return self._result


class _FakeNewEntityClient:
    def __init__(self, result):
        self._result = result
        self.captured_messages = None

    def with_structured_output(self, model):
        return self

    def invoke(self, messages):
        self.captured_messages = messages
        return self._result


def test_classify_document_sends_identical_system_prompt_to_both_calls(monkeypatch):
    """A single shared system prompt (not two divergent ones) is required for the second
    call's request to share a cacheable prefix with the first."""
    from paperlesslabelagent.core.schemas import MatchModel

    match_result = MatchModel(tags=[], correspondent=None, document_type=None)
    new_entities_model = classification_module.build_new_entities_model(
        include_tags=True, include_correspondent=True, include_document_type=True
    )
    new_entity_result = new_entities_model(new_tags=[], new_correspondent=None, new_document_type=None)

    fake_matcher = _FakeMatcherClient(match_result)
    fake_new_entity_proposer = _FakeNewEntityClient(new_entity_result)
    monkeypatch.setattr(classification_module, "matcher", fake_matcher)
    monkeypatch.setattr(classification_module, "new_entity_proposer", fake_new_entity_proposer)

    classification_module.classify_document(
        "sample.pdf",
        "Some document text.",
        _existing_entities(),
        rejected_existing_tags=[],
        rejected_existing_correspondent=None,
        rejected_existing_document_type=None,
        rejected_new_tags=[],
        rejected_new_correspondent=None,
        rejected_new_document_type=None,
    )

    match_system_message = fake_matcher.captured_messages[0]["content"]
    new_entity_system_message = fake_new_entity_proposer.captured_messages[0]["content"]
    assert match_system_message == new_entity_system_message
