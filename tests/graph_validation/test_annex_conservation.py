"""Annex text must be inventoried, or losing it is invisible.

The article and recital inventory keys on ``p.oj-normal``. Annex VI carries no such class
anywhere, so extending the inventory naively would count zero fragments there, reconstruct
zero, and let the gate pass on an annex that never reached the graph. Every assertion here
is therefore per annex, never over the total.
"""
from __future__ import annotations

import pathlib

import pytest
from bs4 import BeautifulSoup

from legal_assistant import config
from legal_assistant.graph.recorder import RecordingGraph
from legal_assistant.validation import act_source as source
from legal_assistant.validation.checks import conservation, normalise
from legal_assistant.validation.plan import GraphPlan

AI_ACT = "32024R1689"
ANNEXLESS = ("32016R0679", "32022R0868", "32023R2854")


def _html_path(celex: str) -> pathlib.Path:
    path = pathlib.Path(config.CORPUS_DIR) / f"{celex}.html"
    if not path.is_file():
        pytest.skip(f"{celex}.html not in corpus/")
    return path


def _annex_divs(celex: str):
    soup = BeautifulSoup(_html_path(celex).read_text(encoding="utf-8"), "html.parser")
    return soup.find_all("div", id=lambda i: i and i.startswith("anx_"))


def _longest_passage(annex_div) -> str:
    """The annex's longest leaf passage, whatever tag or class carries it.

    Deliberately independent of the parser: if the inventory cannot account for an annex's
    single largest piece of prose, that annex is not really covered.

    Only elements holding no nested table count. A cell that contains its sub-points holds
    their text too, and the inventory splits those into fragments of their own, so such a
    cell is a concatenation no single fragment is expected to match.

    Headings are skipped as well. By design they are metadata, carried as ``Annex.title`` and
    ``AnnexPoint.section_heading``, so they are not point text and are not inventoried. In
    Annex XII the title is the longest element there is.
    """
    headings = {"oj-doc-ti", "oj-ti-grseq-1"}
    texts = [
        element.get_text(separator=" ", strip=True)
        for element in annex_div.find_all(["p", "td"])
        if element.find("table") is None
        and not headings & set(element.get("class") or [])
    ]
    return max(texts, key=len)


@pytest.fixture(scope="module")
def ai_act_inventory():
    return source.html_fragments(_html_path(AI_ACT))


@pytest.mark.parametrize("index", range(13))
def test_every_annex_reaches_the_inventory(ai_act_inventory, index):
    """One case per annex: an aggregate count would hide an annex contributing nothing."""
    divs = _annex_divs(AI_ACT)
    annex = divs[index]
    blob = "".join(normalise(fragment) for fragment in ai_act_inventory)

    passage = normalise(_longest_passage(annex))

    assert passage in blob, f"{annex.get('id')} is absent from the source inventory"


def test_a_lost_annex_point_is_reported():
    """The check must fail when a point the source carries never reaches the graph."""
    def build(graph):
        graph.upsert_graph_node("AnnexPoint", {"id": "Xanx_I.p_001", "text": "kept text"})

    plan = GraphPlan.from_recorder(_record(build))
    violations = conservation(
        ["kept text", "dropped text"], source.reconstructed_fragments(plan), kind="act_text"
    )

    assert [v.kind for v in violations] == ["act_text_lost"]


def test_annex_points_count_as_reconstructed():
    """An AnnexPoint is graph text like a Paragraph, so it satisfies the source fragment."""
    def build(graph):
        graph.upsert_graph_node("AnnexPoint", {"id": "Xanx_I.p_001", "text": "the only text"})

    plan = GraphPlan.from_recorder(_record(build))

    assert conservation(["the only text"], source.reconstructed_fragments(plan)) == []


@pytest.mark.parametrize("celex", ANNEXLESS)
def test_annexless_acts_contribute_no_annex_fragments(celex):
    """GDPR, the DGA and the Data Act have no annexes; their inventory must not move."""
    assert source.annex_fragments(_html_path(celex)) == []


def _record(build):
    recorder = RecordingGraph()
    build(recorder)
    return recorder
