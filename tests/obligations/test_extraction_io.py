"""Read the analysis stage's real output, pinned from the paper's own validation data.

The fixture is a verbatim LLM output committed from the source repository's
``setup_evaluation`` set, not written from the paper's Listing 2. If the real shape ever
diverges from what the code expects, it diverges here, cheaply, before any ingestion runs.

The record is faithful to that shape: each element (addressees, targets, ...) is a LIST of
{extraction_method, value}, the predicate is a single object with a verb, and one passage can
yield several obligations. Mapping this onto graph nodes and edges is a later step; this one
only parses.
"""
from __future__ import annotations

import json
import pathlib

from legal_assistant.obligations.extraction_io import parse_analysis
from legal_assistant.obligations.models import ExtractionMethod, ObligationType

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "analysis_output_aia_23_3.json"
AI_ACT = "32024R1689"
SOURCE = f"{AI_ACT}_023.003"


def _parsed():
    record = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return parse_analysis(record, source_id=SOURCE, celex=AI_ACT)


def test_one_obligation_is_read_from_the_passage():
    assert len(_parsed()) == 1


def test_the_obligation_id_composes_from_the_source_and_an_ordinal():
    assert _parsed()[0].id == f"{SOURCE}#ob_1"


def test_the_obligation_type_is_mapped():
    assert _parsed()[0].obligation_type == ObligationType.ACTION


def test_the_predicate_is_read_with_its_voice():
    predicate = _parsed()[0].predicate
    assert predicate.value == "shall indicate"
    assert predicate.verb == "active"
    assert predicate.method == ExtractionMethod.STATED


def test_addressees_are_a_list_of_extracted_elements():
    addressees = _parsed()[0].addressees
    assert [a.value for a in addressees] == ["Importers"]
    assert addressees[0].method == ExtractionMethod.STATED


def test_background_knowledge_is_mapped_from_its_hyphenated_form():
    """The JSON writes 'Background-Knowledge'; our enum is BACKGROUND."""
    beneficiaries = _parsed()[0].beneficiaries
    assert beneficiaries[0].method == ExtractionMethod.BACKGROUND


def test_a_none_element_carries_a_null_value():
    """Pre-Conditions here is a single element with method None and value null."""
    preconditions = _parsed()[0].preconditions
    assert [p.method for p in preconditions] == [ExtractionMethod.NONE]
    assert preconditions[0].value is None
