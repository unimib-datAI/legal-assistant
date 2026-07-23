"""The analysis adapter: drive the paper's analysis prompt, parse into AnalysedObligation.

The prompt already dictates the exact JSON shape, and ``parse_analysis`` already reads it
(pinned against a real output in test_extraction_io). This adapter is the seam between the
two: fill the prompt, call the model, hand the JSON to the reader.
"""
from __future__ import annotations

import json
import pathlib

from legal_assistant.obligations.analysis import analyse_sentence
from legal_assistant.obligations.models import ExtractionMethod, ObligationType

AI_ACT = "32024R1689"
SOURCE = f"{AI_ACT}_023.003"
FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "analysis_output_aia_23_3.json"


class _FakeLLM:
    def __init__(self, content: str):
        self._content = content
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return type("Msg", (), {"content": self._content})()


def _fenced_fixture() -> str:
    return "```json\n" + FIXTURE.read_text(encoding="utf-8") + "\n```"


def test_the_real_output_parses_into_obligations():
    llm = _FakeLLM(_fenced_fixture())
    obligations = analyse_sentence(
        llm, source_id=SOURCE, celex=AI_ACT,
        sentence="Importers shall indicate their name.", context="",
    )
    assert [o.id for o in obligations] == [f"{SOURCE}#ob_1"]
    assert obligations[0].obligation_type == ObligationType.ACTION
    assert obligations[0].predicate.value == "shall indicate"
    assert obligations[0].beneficiaries[0].method == ExtractionMethod.BACKGROUND


def test_the_sentence_and_citations_reach_the_prompt():
    llm = _FakeLLM("```json\n[]\n```")
    analyse_sentence(
        llm, source_id=SOURCE, celex=AI_ACT,
        sentence="The provider shall keep the documentation.",
        context="ctx", citations=["Article 11: the documentation ..."],
    )
    prompt = " ".join(m.content for m in llm.last_messages)
    assert "The provider shall keep the documentation." in prompt
    assert "Article 11: the documentation ..." in prompt


def test_an_empty_array_yields_no_obligations():
    llm = _FakeLLM("```json\n[]\n```")
    assert analyse_sentence(llm, source_id=SOURCE, celex=AI_ACT, sentence="s", context="c") == []


def test_an_unparseable_response_yields_no_obligations():
    """A malformed response must not crash the batch; the sentence simply yields nothing."""
    llm = _FakeLLM("I could not analyse that.")
    assert analyse_sentence(llm, source_id=SOURCE, celex=AI_ACT, sentence="s", context="c") == []
