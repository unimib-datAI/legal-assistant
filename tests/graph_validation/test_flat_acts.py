"""Shapes of act the corpus does not contain, but the acquis does.

Three of them, all covered by the same synthetic fixture because they occur together in
real acts: an act with no chapter division, an annex numbered in Arabic digits, and an
annex that *is* a table rather than a list of numbered points.

Each was a silent loss before: the articles of a flat act never reached the graph, an
``anx_1`` division was never read, and a table contributed only its first row. Silent is
the operative word, which is why they are asserted here rather than left to the gate.
"""
from __future__ import annotations

import pathlib
from unittest.mock import patch

import pytest

from legal_assistant.graph.loader import GraphLoader
from legal_assistant.scraper.eurlex_exporter import EURLexHTMLParser
from legal_assistant.validation import act_source as source
from legal_assistant.validation.checks import conservation, normalise, structural
from legal_assistant.validation.gate import build_plan

CELEX = "39999R0001"
FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "flat_act.html"


@pytest.fixture(scope="module")
def plan():
    """Build the fixture into a recorder. The case-law lookup is the only network call."""
    with patch(
        "legal_assistant.scraper.eurlex_exporter.EURLexHTMLParser._get_case_law",
        return_value=[],
    ):
        data = EURLexHTMLParser(
            html_file_path=str(FIXTURE), celex=CELEX,
            eurolex_url="", document_info_url="",
        ).extract_data()
    return build_plan(lambda graph: GraphLoader(graph)._emit(data))


def _labels(plan, label):
    return [node for node in plan.node_ops if node.label == label]


def test_articles_of_a_flat_act_reach_the_graph(plan):
    """No chapters, so the articles hang off the act. Without this they were dropped."""
    assert len(_labels(plan, "Article")) == 2
    assert not _labels(plan, "Chapter")


def test_flat_articles_are_contained_by_the_act(plan):
    edges = {
        (e.left_label, e.rel_type, e.right_label)
        for e in plan.edge_ops
        if e.right_label == "Article"
    }
    assert edges == {("Act", "CONTAINS", "Article")}


def test_arabic_annex_numbering_is_read(plan):
    """``anx_1`` is as valid as ``anx_III``; matching only Roman numerals lost the annex."""
    assert len(_labels(plan, "Annex")) == 1
    assert _labels(plan, "AnnexPoint")


def test_every_row_of_an_annex_table_is_kept(plan):
    """A table is not one point: each of its rows carries published text."""
    stored = " ".join(
        normalise(node.properties.get("text", "") or "")
        + normalise(node.properties.get("point_label", "") or "")
        for node in _labels(plan, "AnnexPoint")
    )
    for fragment in ("Cauliflowers", "10,52", "Tomatoes", "7,25", "Third country of origin"):
        assert normalise(fragment) in stored, fragment


def test_the_act_is_structurally_sound_and_loses_nothing(plan):
    assert structural(plan, CELEX) == []
    assert conservation(
        source.html_fragments(FIXTURE),
        source.reconstructed_fragments(plan),
        kind="act_text",
    ) == []


def test_the_build_is_deterministic(plan):
    with patch(
        "legal_assistant.scraper.eurlex_exporter.EURLexHTMLParser._get_case_law",
        return_value=[],
    ):
        data = EURLexHTMLParser(
            html_file_path=str(FIXTURE), celex=CELEX,
            eurolex_url="", document_info_url="",
        ).extract_data()
    assert build_plan(lambda graph: GraphLoader(graph)._emit(data)).fingerprint() \
        == plan.fingerprint()
