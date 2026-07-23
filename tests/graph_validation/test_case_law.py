"""Case law must survive both transformations intact, on real judgments.

Three shapes on purpose: a 2012 judgment (the oldest manifestation Cellar serves), a modern
one, and a large one.

The XHTML is **cached, not committed**: 764 KB of source documents does not belong in the
repository. The first run downloads each judgment into ``fixtures/`` (gitignored) and every
run after that is offline. Without network and without a cache, these tests skip rather
than fail: they check the parser, not the availability of Cellar.
"""
from __future__ import annotations

import pathlib
import re
from unittest.mock import MagicMock

import pytest

from legal_assistant.case_law.html_parser import parse_case_law
from legal_assistant.case_law.kg_builder import _write_case_law_kg, create_case_law_kg
from legal_assistant.case_law.tree import flatten
from legal_assistant.validation import case_law_source as source
from legal_assistant.validation.checks import conservation, structural
from legal_assistant.validation.gate import GraphValidationError, build_plan

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
CELEXES = ("62012CJ0293", "62019CJ0645", "62018CJ0511")

_ARTICLE_HEADING = re.compile(r"^Article \d")


def _cached_html(celex: str) -> str:
    """The judgment's XHTML, downloaded once into the gitignored cache."""
    path = FIXTURES / f"{celex}.xhtml"
    if path.is_file():
        return path.read_text(encoding="utf-8")

    from legal_assistant.case_law.html_parser import fetch_html

    try:
        html = fetch_html(celex)
    except Exception as exc:  # offline, Cellar down, proxy, not a parser failure
        pytest.skip(f"{celex} not cached and could not be fetched: {exc}")

    FIXTURES.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return html


@pytest.fixture(scope="module", params=CELEXES)
def judgment(request):
    celex = request.param
    html = _cached_html(celex)
    roots = parse_case_law(html)
    flat = flatten(roots)
    plan = build_plan(lambda g: _write_case_law_kg(celex, flat, g))
    return celex, html, flat, plan


def test_layer1_html_to_tree_loses_nothing(judgment):
    """Every fragment the parser reads must appear in the tree, bar the index block."""
    _, html, flat, _ = judgment
    violations = conservation(
        source.html_fragments(html),
        source.tree_fragments(flat),
        exempt=source.html_exemptions(html),
        kind="html",
    )
    assert violations == [], "\n".join(str(v) for v in violations)


def test_layer2_tree_to_graph_loses_nothing(judgment):
    """Every body item of a substantive section must land in a CaseLawParagraph."""
    _, _, flat, plan = judgment
    violations = conservation(
        source.body_fragments(flat),
        source.paragraph_texts(plan),
        exempt=source.body_exemptions(flat),
        kind="body",
    )
    assert violations == [], "\n".join(str(v) for v in violations)


def test_structure_is_well_formed(judgment):
    celex, _, _, plan = judgment
    violations = structural(plan, celex)
    assert violations == [], "\n".join(str(v) for v in violations)


def test_section_depth_never_exceeds_two(judgment):
    _, _, _, plan = judgment
    depths = [n.properties["depth"] for n in plan.node_ops if n.label == "CaseLawSection"]
    assert depths and max(depths) <= 2


def test_no_article_headings_leak_into_sections(judgment):
    """An 'Article 5' heading means legislation was mistaken for judgment structure."""
    _, _, _, plan = judgment
    headings = [n.properties["heading"] for n in plan.node_ops if n.label == "CaseLawSection"]
    assert [h for h in headings if _ARTICLE_HEADING.match(h)] == []


def test_topics_section_becomes_topic_nodes_not_a_section(judgment):
    _, _, flat, plan = judgment
    headings = [s["heading"] for s in flat]
    assert headings.count("Topics") == 1, "the parser must emit exactly one Topics section"

    section_headings = [n.properties["heading"] for n in plan.node_ops
                        if n.label == "CaseLawSection"]
    assert "Topics" not in section_headings, "Topics must not become a section node"
    assert [n for n in plan.node_ops if n.label == "CaseLawTopic"], "no topic nodes emitted"


def test_parsing_is_deterministic(judgment):
    celex, html, _, plan = judgment
    again = build_plan(lambda g: _write_case_law_kg(celex, flatten(parse_case_law(html)), g))
    assert again.fingerprint() == plan.fingerprint()


def test_create_case_law_kg_writes_a_valid_judgment(judgment):
    """The public entry point validates and then writes."""
    celex, _, flat, plan = judgment
    graph = MagicMock()
    counts = create_case_law_kg(celex=celex, flat=flat, graph=graph)

    assert graph.upsert_graph_node.call_count == len(plan.node_ops)
    assert counts["paragraphs"] == sum(1 for n in plan.node_ops
                                       if n.label == "CaseLawParagraph")


def test_create_case_law_kg_writes_nothing_when_the_splitter_drops_a_paragraph(
    judgment, monkeypatch
):
    """Simulate the regression the gate exists to catch: the splitter silently loses text.

    Corrupting ``flat`` would not do it: the inventory is derived from ``flat`` too, so both
    sides move together. The asymmetry has to come from the builder, which is exactly where
    a real parser regression would live.
    """
    celex, _, flat, _ = judgment
    import legal_assistant.case_law.kg_builder as kg

    real_split = kg.split_paragraphs

    def lossy_split(celex_, body):
        paragraphs = real_split(celex_, body)
        return paragraphs[:-1] if len(paragraphs) > 1 else paragraphs

    monkeypatch.setattr(kg, "split_paragraphs", lossy_split)

    graph = MagicMock()
    with pytest.raises(GraphValidationError, match="body_lost"):
        create_case_law_kg(celex=celex, flat=flat, graph=graph)

    graph.upsert_graph_node.assert_not_called()
    graph.create_relationship.assert_not_called()


def test_operative_part_is_marked(judgment):
    """A judgment always rules on something; the holding must be flagged as operative."""
    _, _, _, plan = judgment
    operative = [n for n in plan.node_ops
                 if n.label == "CaseLawParagraph" and n.properties["is_operative"]]
    assert operative, "no operative paragraph found"
    assert all("_op_" in n.id for n in operative)
