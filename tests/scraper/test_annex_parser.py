"""The annex splitter, run against the real AI Act HTML in ``corpus/``.

Annexes are binding text: Annex III decides whether Chapter III applies at all, and Annex IV
is the technical documentation a provider owes under Article 11(1). Only the AI Act has any.

The published markup offers no nested element per point, so these tests pin the two shapes
that do occur, and in particular Annex VI, which carries no ``oj-normal`` class at all and is
therefore invisible to any splitter keyed on it.
"""
from __future__ import annotations

import pathlib
from unittest.mock import patch

import pytest

from legal_assistant import config
from legal_assistant.scraper.eurlex_exporter import EURLexHTMLParser

AI_ACT = "32024R1689"


def _parser(celex: str) -> EURLexHTMLParser:
    path = pathlib.Path(config.CORPUS_DIR) / f"{celex}.html"
    if not path.is_file():
        pytest.skip(f"{celex}.html not in corpus/")
    return EURLexHTMLParser(
        html_file_path=str(path),
        celex=celex,
        eurolex_url=f"https://eur-lex.europa.eu/eli/{celex}",
        document_info_url="",
    )


@pytest.fixture(scope="module")
def ai_act_annexes():
    return _parser(AI_ACT)._get_annexes()


def _by_number(annexes, number):
    return next(annex for annex in annexes if annex["number"] == number)


def _numbered(annex):
    """The points that carry a marker, dropping unnumbered lead-in prose."""
    return [point for point in annex["points"] if point["label"] is not None]


def test_ai_act_has_thirteen_annexes(ai_act_annexes):
    numbers = [annex["number"] for annex in ai_act_annexes]
    assert numbers == [
        "I", "II", "III", "IV", "V", "VI", "VII",
        "VIII", "IX", "X", "XI", "XII", "XIII",
    ]


def test_annex_title_excludes_the_annex_number(ai_act_annexes):
    """"ANNEX III" is the number, already carried by `number`; the title is what follows."""
    assert _by_number(ai_act_annexes, "III")["title"] == (
        "High-risk AI systems referred to in Article\xa06(2)"
    )


def test_annex_vi_yields_its_four_points(ai_act_annexes):
    """The shape with no ``oj-normal`` class anywhere.

    Annex VI states the internal-control conformity assessment procedure as four unclassed
    ``<p>`` elements alternating number and prose. A splitter keyed on ``oj-normal``, which
    is what the article and recital side uses, finds nothing here and silently yields an
    empty annex.
    """
    points = _by_number(ai_act_annexes, "VI")["points"]
    assert [point["label"] for point in points] == ["1", "2", "3", "4"]


def test_annex_vi_keeps_label_and_prose_apart(ai_act_annexes):
    """A point's text is the prose, never the number glued onto it."""
    second = _by_number(ai_act_annexes, "VI")["points"][1]
    assert second["text"].startswith("The provider verifies")


def test_annex_iii_reads_the_table_shape(ai_act_annexes):
    """The other shape: each numbered point is a ``td`` pair, label cell then text cell."""
    first = _numbered(_by_number(ai_act_annexes, "III"))[0]
    assert first["label"] == "1"
    assert first["text"].startswith("Biometrics")


def test_annex_iii_keeps_its_lead_in_sentence(ai_act_annexes):
    """Prose outside any table is still published text and must reach the graph.

    Annex III opens with the sentence that governs everything under it, "High-risk AI
    systems pursuant to Article 6(2) are the AI systems listed in any of the following
    areas". It sits outside the tables and carries no marker of its own, so it is a point
    with a null label rather than something to drop.
    """
    texts = [point["text"] for point in _by_number(ai_act_annexes, "III")["points"]]
    assert any("listed in any of the following areas" in text for text in texts)


def test_indented_points_find_their_marker(ai_act_annexes):
    """Annex VII indents with an empty leading cell, so the marker is not the first cell.

    Its rows are three cells wide, ``['', '3.1.', 'The application ...']``. Reading the
    marker off cell zero yields an empty label and pushes the real marker into the text.
    """
    point = next(
        p for p in _numbered(_by_number(ai_act_annexes, "VII")) if p["label"] == "3.1"
    )
    assert point["text"].startswith("The application of the provider")


def test_points_carry_their_section_heading(ai_act_annexes):
    """Annex VII groups its points under headings, denormalised onto each point.

    Same reason ``CaseLawParagraph`` carries ``section_heading``: at retrieval time the
    heading is context the point itself does not state.
    """
    point = next(
        p for p in _numbered(_by_number(ai_act_annexes, "VII")) if p["label"] == "3.1"
    )
    assert point["section_heading"] == "3.\xa0\xa0\xa0Quality management system"


def test_headings_are_not_points(ai_act_annexes):
    """A heading is navigation, not a provision, so it must not become retrievable text."""
    texts = [point["text"] for point in _by_number(ai_act_annexes, "VII")["points"]]
    assert "3.\xa0\xa0\xa0Quality management system" not in texts


def test_annex_without_headings_has_none(ai_act_annexes):
    """Annex III has no internal headings; its points say so rather than inventing one."""
    headings = {point["section_heading"] for point in _by_number(ai_act_annexes, "III")["points"]}
    assert headings == {None}


def test_annex_iii_sub_points_compose_their_label(ai_act_annexes):
    """Sub-points are nested tables, and a point's label carries its whole path.

    Article 6(2) points at "Annex III, point 1(a)", so the path has to survive parsing;
    the annex number is prepended later, by the loader that knows it.
    """
    labels = [point["label"] for point in _numbered(_by_number(ai_act_annexes, "III"))]
    assert labels[:4] == ["1", "1(a)", "1(b)", "1(c)"]


def test_extract_data_exposes_annexes():
    """The loader reads the parser's dict, so annexes have to be in it to reach the graph."""
    with patch.object(EURLexHTMLParser, "_get_case_law", return_value=[]):
        data = _parser(AI_ACT).extract_data()

    assert [annex["number"] for annex in data["annexes"]][:3] == ["I", "II", "III"]


@pytest.mark.parametrize("celex", ["32016R0679", "32022R0868", "32023R2854"])
def test_acts_without_annexes_yield_none(celex):
    """GDPR, the DGA and the Data Act have no annexes at all, and must stay untouched."""
    assert _parser(celex)._get_annexes() == []


def test_unknown_shape_keeps_its_text(tmp_path):
    """An annex matching neither shape yields unlabelled points, never nothing.

    Losing text silently is the failure the validation gate exists to catch, so the splitter
    degrades to "no label" rather than to "no point".
    """
    html = (
        '<div id="anx_XIV">'
        '<p class="oj-doc-ti">ANNEX XIV</p>'
        '<p class="oj-doc-ti">Some future annex</p>'
        '<p class="oj-normal">Prose carrying no marker of any kind.</p>'
        "</div>"
    )
    path = tmp_path / "future.html"
    path.write_text(html, encoding="utf-8")

    annexes = EURLexHTMLParser(str(path), "X", "", "")._get_annexes()

    assert [(p["label"], p["text"]) for p in annexes[0]["points"]] == [
        (None, "Prose carrying no marker of any kind.")
    ]
