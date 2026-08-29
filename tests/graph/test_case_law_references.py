"""Interpretation edges must point at a node that exists, or not be emitted at all.

The real ``CREATE_RELATIONSHIP`` is ``MATCH ... MATCH ... MERGE``, so an edge whose endpoint
is missing writes nothing and raises nothing. That silence is how a mistyped reference turns
into a judgment the ingest never sees: ``GET_CASE_LAW_BY_ACTS`` finds judgments through
INTERPRETS, so no edge means no ingest, forever and without a message.
"""
from __future__ import annotations

import pytest

from legal_assistant.graph.loader import GraphLoader
from legal_assistant.graph.recorder import RecordingGraph
from legal_assistant.validation.checks import dangling_edges
from legal_assistant.validation.plan import GraphPlan

CELEX = "32016R0679"


def _act(paragraph_ids: list[str], case_law: list[dict]) -> GraphPlan:
    """One article with the given paragraph ids, plus the case law references."""
    data = {
        "act": {"celex": CELEX, "title": "t", "eurolex_url": ""},
        "chapters": [{
            "id": "cpt_I", "title": "C", "sections": [],
            "articles": [{
                "id": "art_4", "number": "Article 4", "title": "Definitions",
                "full_text": "x",
                "paragraphs": [{"id": pid, "text": "p"} for pid in paragraph_ids],
            }],
        }],
        "recitals": [], "annexes": [], "case_law": case_law,
    }
    recorder = RecordingGraph()
    GraphLoader(recorder)._emit(data)
    return GraphPlan.from_recorder(recorder)


def _edges(plan: GraphPlan) -> list[tuple[str, str]]:
    return [(e.left_id, e.right_id) for e in plan.edge_ops if e.rel_type == "INTERPRETS"]


def test_a_padded_paragraph_reference_resolves():
    plan = _act(["006.001"], [{"case_law_identifier": "C1", "article": "art_6",
                               "paragraph": "006.001"}])
    assert _edges(plan) == [("C1", f"{CELEX}_006.001")]


def test_a_definition_point_falls_back_to_the_unpadded_id():
    """EUR-Lex writes both 'A04PT7' and 'A04P4'; the loader stores definitions unpadded."""
    plan = _act(["004.4"], [{"case_law_identifier": "C1", "article": "art_4",
                             "paragraph": "004.004"}])
    assert _edges(plan) == [("C1", f"{CELEX}_004.4")]


def test_an_unresolvable_paragraph_falls_back_to_the_article():
    """Better a coarser true edge than no edge: the judgment stays reachable."""
    plan = _act(["004.1"], [{"case_law_identifier": "C1", "article": "art_4",
                             "paragraph": "004.099"}])
    assert _edges(plan) == [("C1", f"{CELEX}art_4")]


def test_a_reference_with_no_usable_target_emits_no_edge(caplog):
    plan = _act(["004.1"], [{"case_law_identifier": "C1"}])
    with caplog.at_level("WARNING"):
        plan = _act(["004.1"], [{"case_law_identifier": "C1"}])
    assert _edges(plan) == []
    assert "C1" in caplog.text


def test_the_stub_is_created_even_when_the_reference_does_not_resolve():
    """Losing the judgment entirely would be worse than knowing of it without a target."""
    plan = _act(["004.1"], [{"case_law_identifier": "C1"}])
    assert any(n.label == "CaseLaw" and n.id == "C1" for n in plan.node_ops)


@pytest.mark.parametrize("case_law", [
    [{"case_law_identifier": "C1", "article": "art_4", "paragraph": "004.004"}],
    [{"case_law_identifier": "C1", "article": "art_4", "paragraph": "004.099"}],
    [{"case_law_identifier": "C1"}],
])
def test_no_interpretation_edge_ever_dangles(case_law):
    """With INTERPRETS no longer exempt, this is what the gate now enforces on every act."""
    plan = _act(["004.4"], case_law)
    assert dangling_edges(plan) == []
