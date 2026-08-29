"""An interrupted build must resume, not restart.

The assertion that matters throughout: an act already in the graph is never *downloaded*.
Checking after the fetch would still be correct, but it would pay Playwright and the EUR-Lex
WAF challenge for work already done, which is most of what an interrupted run wastes.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from legal_assistant.pipelines.graph_build import build_graph

GDPR = "32016R0679"
AI_ACT = "32024R1689"


def _config(celex: str) -> dict:
    return {"html_file": f"/nonexistent/{celex}.html", "celex": celex,
            "eurolex_url": "", "document_info_url": ""}


@pytest.fixture
def build(graph_mock):
    """Run ``build_graph`` with the network, the loader and the embeddings stubbed out.

    Yields a callable plus the mocks, so each test asserts on what was fetched and written
    rather than on what was returned.
    """
    with patch("legal_assistant.pipelines.graph_build.make_graph_client", return_value=graph_mock), \
         patch("legal_assistant.pipelines.graph_build.EurlexDocumentUtils") as utils, \
         patch("legal_assistant.pipelines.graph_build.GraphLoader") as loader_cls, \
         patch("legal_assistant.pipelines.graph_build.make_embeddings"):
        utils.return_value.build_document_config.side_effect = _config
        loader = loader_cls.return_value
        loader.plan_document.side_effect = lambda config, **kw: MagicMock(name=config["celex"])
        loader.plan_all_documents.side_effect = lambda configs, **kw: [
            (c["celex"], MagicMock()) for c in configs
        ]
        graph_mock.generate_text_embeddings.return_value = 1536

        yield lambda *a, **kw: build_graph(*a, **kw), utils.return_value, loader, graph_mock


def _loaded(graph_mock, *celex: str) -> None:
    """Make the resume predicate answer yes for ``celex`` and no for anything else."""
    graph_mock.query.side_effect = lambda _q, params: [{"done": params["celex"] in celex}]


def test_an_act_already_present_is_never_downloaded(build):
    run, utils, loader, graph = build
    _loaded(graph, GDPR)

    result = run([GDPR])

    utils.build_document_config.assert_not_called()
    loader.plan_document.assert_not_called()
    loader.write.assert_not_called()
    assert result.skipped == [GDPR]
    assert result.celex_ids == []


def test_an_absent_act_is_fetched_and_written(build):
    run, utils, loader, graph = build
    _loaded(graph)  # nothing is loaded

    result = run([GDPR])

    utils.build_document_config.assert_called_once_with(GDPR)
    loader.write.assert_called_once()
    assert result.celex_ids == [GDPR]
    assert result.skipped == []


def test_a_mixed_batch_processes_only_what_is_missing(build):
    run, utils, loader, graph = build
    _loaded(graph, GDPR)

    result = run([GDPR, AI_ACT])

    assert [c.args[0] for c in utils.build_document_config.call_args_list] == [AI_ACT]
    assert result.celex_ids == [AI_ACT]
    assert result.skipped == [GDPR]


def test_force_reprocesses_an_act_that_is_already_present(build):
    run, utils, loader, graph = build
    _loaded(graph, GDPR)

    result = run([GDPR], force=True)

    utils.build_document_config.assert_called_once_with(GDPR)
    assert result.celex_ids == [GDPR]
    assert result.skipped == []


def test_the_database_is_not_cleared_by_default(build):
    run, _, _, graph = build
    _loaded(graph)

    run([GDPR])

    graph.clear_database.assert_not_called()


def test_clear_wipes_and_consults_no_predicate(build):
    """After a wipe every unit is absent by definition, so asking would be noise."""
    run, utils, loader, graph = build
    _loaded(graph, GDPR)  # would say "skip" if it were consulted

    result = run([GDPR], clear_db=True)

    graph.clear_database.assert_called_once()
    graph.query.assert_not_called()
    loader.plan_all_documents.assert_called_once()
    assert result.celex_ids == [GDPR]
    assert result.skipped == []


def test_clear_still_validates_every_act_before_wiping(build):
    """The pre-existing guarantee: a parser regression must not cost a good graph."""
    run, _, loader, graph = build
    order = []
    graph.clear_database.side_effect = lambda: order.append("clear")
    loader.plan_all_documents.side_effect = lambda configs, **kw: (
        order.append("plan") or [(c["celex"], MagicMock()) for c in configs]
    )
    loader.write.side_effect = lambda plan: order.append("write")

    run([GDPR, AI_ACT], clear_db=True)

    assert order == ["plan", "clear", "write", "write"]


def test_a_broken_act_does_not_block_the_others_when_resuming(build):
    """Per-act writes mean act 2 still lands when act 1 fails to parse."""
    run, _, loader, graph = build
    _loaded(graph)

    def plan(config, **kw):
        if config["celex"] == GDPR:
            raise RuntimeError("parser regression")
        return MagicMock()

    loader.plan_document.side_effect = plan
    result = run([GDPR, AI_ACT])

    assert result.celex_ids == [AI_ACT]
    assert result.failed == [(GDPR, "parser regression")]


def test_the_connection_is_closed_even_when_everything_is_skipped(build):
    run, _, _, graph = build
    _loaded(graph, GDPR)

    run([GDPR])

    graph.close.assert_called_once()
