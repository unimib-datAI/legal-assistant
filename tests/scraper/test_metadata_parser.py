"""EUR-Lex "Interpreted by" references, normalised to graph coordinates.

The reference language is finer-grained than the graph: EUR-Lex can cite a sentence inside a
subparagraph inside a point, while the graph stops at the paragraph. Everything below a
paragraph is therefore truncated, never encoded into the number, which is the failure these
tests exist to pin: reading *all* the digits out of "A06P1L1LF" yields paragraph 11 of an
article that has four.
"""
from __future__ import annotations

import pytest

from legal_assistant.scraper.metadata_parser import MetadataParser

REF = MetadataParser.enrich_article_reference


@pytest.mark.parametrize("reference,expected", [
    ("A01", (None, "art_1", None)),
    ("A02P1", (None, "art_2", "002.001")),
    ("CH8", ("CH8", None, None)),
])
def test_the_plain_shapes(reference, expected):
    assert REF(reference) == expected


@pytest.mark.parametrize("reference,paragraph", [
    ("A06P1LA", "006.001"),      # point (a) of paragraph 1
    ("A06P1L1LF", "006.001"),    # first subparagraph, point (f)
    ("A22P3L2", "022.003"),      # second subparagraph
    ("A38P3SNT2", "038.003"),    # second sentence
    ("A15P3SNT1", "015.003"),
    ("A05P1LA-LF", "005.001"),   # a range of points
    ("A83P5LB", "083.005"),
])
def test_subdivisions_below_the_paragraph_are_truncated_not_absorbed(reference, paragraph):
    """The digits after the paragraph number belong to a finer unit, not to the number."""
    assert REF(reference)[2] == paragraph


@pytest.mark.parametrize("reference,paragraph", [
    ("A04PT7", "004.7"),
    ("A04PT11", "004.11"),
    ("A04PT1", "004.1"),
])
def test_definition_points_use_the_loader_id_shape(reference, paragraph):
    """Split definitions are stored unpadded (``004.7``), unlike numbered paragraphs.

    Producing the padded ``004.007`` here is what made every Article 4 reference miss: the
    node exists, under the other spelling.
    """
    assert REF(reference)[2] == paragraph


@pytest.mark.parametrize("reference", ["A12-A15", "A12-A15P1"])
def test_a_range_of_articles_does_not_raise(reference):
    """``int('12-A15')`` used to propagate out and abort the whole act build."""
    assert REF(reference) == (None, None, None)


@pytest.mark.parametrize("reference", ["N", "PR", "L", "", "ANN1"])
def test_an_unrecognised_shape_yields_no_target(reference):
    assert REF(reference) == (None, None, None)


def test_an_unrecognised_shape_is_logged(caplog):
    """Silence here is how twelve judgments went missing without anyone noticing."""
    with caplog.at_level("WARNING"):
        REF("PR")
    assert "PR" in caplog.text
