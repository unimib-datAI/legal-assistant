"""The filtering adapter: drive the paper's filtering prompt with our LLM client.

The prompt is the paper's validated asset, used unchanged; this adapter only supplies the
sentence, context and citations, calls the model, and reads the ``{classification,
justification}`` JSON it returns. Only "Deontic obligation" and "Deontic prohibition" advance
to analysis, as in the paper.
"""
from __future__ import annotations

from legal_assistant.obligations.filtering import (
    FilteringCategory,
    filter_sentence,
    is_deontic,
)


class _FakeLLM:
    """A chat model stand-in that returns a canned string, and records the prompt it saw."""

    def __init__(self, content: str):
        self._content = content
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return type("Msg", (), {"content": self._content})()


def _json(classification: str) -> str:
    return (
        "```json\n{\n"
        f'  "classification": "{classification}",\n'
        '  "justification": "because"\n'
        "}\n```"
    )


def test_a_deontic_obligation_is_classified():
    llm = _FakeLLM(_json("Deontic obligation"))
    result = filter_sentence(llm, sentence="Importers shall indicate their name.", context="")
    assert result.classification == FilteringCategory.DEONTIC_OBLIGATION


def test_the_sentence_and_context_reach_the_prompt():
    """The adapter substitutes into the paper's user template, not a prompt of its own."""
    llm = _FakeLLM(_json("Deontic obligation"))
    filter_sentence(llm, sentence="X shall do Y.", context="Surrounding paragraph.")
    prompt = " ".join(m.content for m in llm.last_messages)
    assert "X shall do Y." in prompt
    assert "Surrounding paragraph." in prompt


def test_absent_citations_render_as_no_citation():
    """The paper's template always has a citations slot; empty means 'No Citation', not blank."""
    llm = _FakeLLM(_json("Definition"))
    filter_sentence(llm, sentence="s", context="c", citations=[])
    prompt = " ".join(m.content for m in llm.last_messages)
    assert "No Citation" in prompt


def test_only_deontic_categories_advance():
    assert is_deontic(FilteringCategory.DEONTIC_OBLIGATION)
    assert is_deontic(FilteringCategory.DEONTIC_PROHIBITION)
    assert not is_deontic(FilteringCategory.DEFINITION)
    assert not is_deontic(FilteringCategory.NOT_APPLICABLE)


def test_an_unrecognised_classification_falls_back_to_not_applicable():
    """A label the model invents must not crash the batch; it just does not advance."""
    llm = _FakeLLM(_json("Something new"))
    result = filter_sentence(llm, sentence="s", context="c")
    assert result.classification == FilteringCategory.NOT_APPLICABLE
    assert not is_deontic(result.classification)
