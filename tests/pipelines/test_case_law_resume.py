"""A judgment already in the graph must not be fetched from Cellar again.

The predicate here is the subtle one in the whole design. Every judgment has a ``(:CaseLaw)``
node from the moment its act was loaded, because the act loader creates a stub for each
"Interpreted by" reference. Asking whether the node exists would answer yes for all of them
and skip the entire corpus, so the question has to be whether it has content.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from legal_assistant.case_law.html_parser import CaseLawHTMLError
from legal_assistant.pipelines.case_law_ingest import ingest

STORED = "62019CJ0645"
MISSING = "62018CJ0511"


@pytest.fixture
def parse():
    with patch("legal_assistant.pipelines.case_law_ingest.parse_celex") as parse_celex, \
         patch("legal_assistant.pipelines.case_law_ingest.build_from_tree") as build, \
         patch("legal_assistant.pipelines.case_law_ingest.flatten", return_value=[]):
        build.return_value = {"sections": 3, "paragraphs": 40, "operative": 2}
        yield parse_celex, build


def _ingested(graph_mock, *celex: str) -> None:
    graph_mock.query.side_effect = lambda _q, params: [{"done": params["celex"] in celex}]


def test_a_stored_judgment_is_never_fetched(parse, graph_mock):
    parse_celex, build = parse
    _ingested(graph_mock, STORED)

    totals = ingest(graph_mock, [STORED])

    parse_celex.assert_not_called()
    build.assert_not_called()
    assert totals.skipped == 1
    assert totals.judgments == 0


def test_a_stub_without_sections_is_fetched(parse, graph_mock):
    """The normal state right after an act build. Getting this wrong skips everything."""
    parse_celex, build = parse
    _ingested(graph_mock)  # the stub exists, but has no sections

    totals = ingest(graph_mock, [MISSING])

    parse_celex.assert_called_once_with(MISSING)
    assert totals.judgments == 1
    assert totals.skipped == 0


def test_a_mixed_batch_fetches_only_what_is_missing(parse, graph_mock):
    parse_celex, _ = parse
    _ingested(graph_mock, STORED)

    totals = ingest(graph_mock, [STORED, MISSING])

    assert [c.args[0] for c in parse_celex.call_args_list] == [MISSING]
    assert totals.judgments == 1
    assert totals.skipped == 1


def test_an_unfetchable_judgment_is_recorded_and_the_batch_continues(parse, graph_mock):
    parse_celex, _ = parse
    _ingested(graph_mock)
    parse_celex.side_effect = [CaseLawHTMLError("no XHTML in Cellar"), None]

    totals = ingest(graph_mock, [MISSING, STORED])

    assert totals.judgments == 1
    assert [celex for celex, _ in totals.failed] == [MISSING]


def test_skipping_costs_one_query_and_no_summaries(parse, graph_mock):
    """Resuming must be cheap, otherwise it is not worth doing before the fetch."""
    _ingested(graph_mock, STORED)

    with patch("legal_assistant.case_law.llm_orchestrator.summarize_section") as summarize:
        ingest(graph_mock, [STORED], with_summaries=True)

    summarize.assert_not_called()
    assert graph_mock.query.call_count == 1
