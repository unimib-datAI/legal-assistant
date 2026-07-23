"""An act must reach the graph without losing a single published fragment.

Runs against the four HTML files already in ``corpus/``, no network. The case-law lookup
inside ``extract_data`` is the one part that would reach out, so it is stubbed: the
"Interpreted by" metadata is a separate concern from whether the act's own text survives.
"""
from __future__ import annotations

import json
import pathlib
from unittest.mock import patch

import pytest

from legal_assistant import config
from legal_assistant.graph.loader import GraphLoader
from legal_assistant.validation import act_source as source
from legal_assistant.validation.checks import conservation, structural
from legal_assistant.validation.gate import build_plan

CELEXES = ("32016R0679", "32022R0868", "32023R2854", "32024R1689")
GOLDEN = pathlib.Path(__file__).parent / "golden_fingerprints.json"


def _html_path(celex: str) -> pathlib.Path:
    return config.CORPUS_DIR / f"{celex}.html"


def _plan_for(celex: str):
    """Build the act into a recorder. Case law metadata (network) is stubbed out."""
    cfg = {
        "html_file": str(_html_path(celex)),
        "celex": celex,
        "eurolex_url": f"https://eur-lex.europa.eu/eli/{celex}",
        "document_info_url": "",
    }
    with patch(
        "legal_assistant.scraper.eurlex_exporter.EURLexHTMLParser._get_case_law",
        return_value=[],
    ):
        return build_plan(lambda g: GraphLoader(g).load_document(cfg))


@pytest.fixture(scope="module", params=CELEXES)
def act(request):
    celex = request.param
    if not _html_path(celex).is_file():
        pytest.skip(f"{celex}.html not in corpus/")
    return celex, _plan_for(celex)


def test_structure_is_well_formed(act):
    celex, plan = act
    violations = structural(plan, celex)
    assert violations == [], "\n".join(str(v) for v in violations)


def test_no_text_is_lost(act):
    """The check this whole effort exists for."""
    celex, plan = act
    violations = conservation(
        source.html_fragments(_html_path(celex)),
        source.reconstructed_fragments(plan),
        kind="act_text",
    )
    assert violations == [], (
        f"{celex}: published text did not reach the graph\n"
        + "\n".join(str(v) for v in violations)
    )


def test_every_article_has_at_least_one_paragraph(act):
    """An article whose paragraphs all vanished is the failure mode that started this."""
    celex, plan = act
    with_paragraphs = {e.left_id for e in plan.edge_ops
                       if e.rel_type == "CONTAINS" and e.right_label == "Paragraph"}
    empty = sorted(op.id for op in plan.node_ops
                   if op.label == "Article" and op.id not in with_paragraphs)
    assert empty == [], f"{celex}: articles with no paragraph: {empty}"


def test_parsing_is_deterministic(act):
    celex, plan = act
    assert _plan_for(celex).fingerprint() == plan.fingerprint()


@pytest.mark.skipif(not GOLDEN.is_file(), reason="golden fingerprints not frozen yet")
def test_fingerprint_matches_golden(act):
    celex, plan = act
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if celex not in golden:
        pytest.skip(f"no golden fingerprint for {celex}")
    assert plan.fingerprint() == golden[celex], (
        f"{celex}: the built graph changed. If intentional, refresh "
        f"{GOLDEN.name} and say why in the commit."
    )
