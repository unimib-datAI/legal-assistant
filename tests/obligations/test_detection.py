"""The graph-driven detector: which passages carry a candidate obligation, and where from.

Replaces the source repository's ``obligation_detection``. Theirs fetches EUR-Lex by URL and
segments on ``div id`` matching ``^\\d+\\.\\d+``; ours draws passages from the graph, which
recovers the 14 AI Act articles with no numbered paragraph div (Article 16 among them),
sidesteps the six colliding amending-article ids the loader already re-derived, and lets
annex points through. Sentence splitting and marker matching are kept.

The marker logic is a pure function over text, so most of this needs no graph.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from legal_assistant.graph.queries import NodeQueries
from legal_assistant.obligations.detection import detect, detect_in_passage

AI_ACT = "32024R1689"


# ── the pure marker/sentence logic ───────────────────────────────────────────

def test_a_sentence_with_a_modal_becomes_a_candidate():
    result = detect_in_passage(
        f"{AI_ACT}_023.003",
        "Importers shall indicate their name on the high-risk AI system.",
        AI_ACT,
    )
    assert [c.sentence for c in result.candidates] == [
        "Importers shall indicate their name on the high-risk AI system."
    ]


def test_a_passage_with_no_modal_yields_no_candidate():
    result = detect_in_passage(
        f"{AI_ACT}_001.001", "This Regulation lays down harmonised rules.", AI_ACT
    )
    assert result.candidates == []


def test_only_the_modal_sentence_is_kept_within_a_passage():
    """A paragraph mixes descriptive and prescriptive sentences; only the latter carry a duty."""
    text = (
        "This Article applies to high-risk AI systems. "
        "The provider shall establish a risk management system. "
        "It covers the whole lifecycle."
    )
    result = detect_in_passage(f"{AI_ACT}_009.001", text, AI_ACT)
    assert [c.sentence for c in result.candidates] == [
        "The provider shall establish a risk management system."
    ]


def test_must_and_have_to_are_markers():
    text = (
        "The system must be robust. "
        "Deployers have to monitor its operation. "
        "The report has to be retained."
    )
    result = detect_in_passage(f"{AI_ACT}_014.001", text, AI_ACT)
    assert len(result.candidates) == 3


def test_a_bare_auxiliary_has_is_not_a_marker():
    """"has to" is a duty; a bare auxiliary "has" is not, or every clause would match."""
    result = detect_in_passage(
        f"{AI_ACT}_003.001", "The Commission has adopted the delegated act.", AI_ACT
    )
    assert result.candidates == []


def test_a_cited_provision_is_attached_to_the_candidate():
    """The analysis stage reads cited provisions; detection records which they are."""
    result = detect_in_passage(
        f"{AI_ACT}_023.003",
        "The provider shall keep the documentation referred to in Article 11.",
        AI_ACT,
    )
    assert f"{AI_ACT}art_11" in result.candidates[0].references


def test_a_cross_act_citation_is_left_to_the_analysis_stage():
    """"Article 11 of Regulation (EU) 2016/679" cites the GDPR, not this act's Article 11."""
    result = detect_in_passage(
        f"{AI_ACT}_010.001",
        "The provider shall comply with Article 11 of Regulation (EU) 2016/679.",
        AI_ACT,
    )
    assert result.candidates[0].references == []


# ── the graph-backed pass ────────────────────────────────────────────────────

def _graph(paragraph_rows, annex_rows=()):
    graph = MagicMock()

    def query(statement, params=None):
        if statement == NodeQueries.GET_PARAGRAPHS_BY_ACTS:
            return list(paragraph_rows)
        if statement == NodeQueries.GET_ANNEX_POINTS_BY_ACTS:
            return list(annex_rows)
        return []

    graph.query.side_effect = query
    return graph


def test_detect_reads_paragraphs_from_the_graph():
    graph = _graph([
        {"id": f"{AI_ACT}_016.000", "text": "The provider shall comply with Article 16.",
         "celex": AI_ACT},
        {"id": f"{AI_ACT}_001.001", "text": "This Regulation lays down rules.",
         "celex": AI_ACT},
    ])
    detected = detect(graph, [AI_ACT])

    ids = [p.par_id for p in detected]
    assert ids == [f"{AI_ACT}_016.000"]


def test_detect_includes_annex_points():
    """Annex IV opens 'shall contain', an obligation invisible to a paragraph-div regex."""
    graph = _graph(
        paragraph_rows=[],
        annex_rows=[{"id": f"{AI_ACT}anx_IV.p_001",
                     "text": "The technical documentation shall contain the following.",
                     "celex": AI_ACT}],
    )
    detected = detect(graph, [AI_ACT])

    assert [p.par_id for p in detected] == [f"{AI_ACT}anx_IV.p_001"]


def test_detect_never_reads_recitals():
    """Recitals do not bind. The detector selects passage nodes; it must not select them."""
    graph = _graph([{"id": f"{AI_ACT}_005.001", "text": "The provider shall not do X.",
                     "celex": AI_ACT}])
    detect(graph, [AI_ACT])

    called = [call.args[0] for call in graph.query.call_args_list]
    assert NodeQueries.GET_PARAGRAPHS_BY_ACTS in called
    assert all("Recital" not in q for q in called)
