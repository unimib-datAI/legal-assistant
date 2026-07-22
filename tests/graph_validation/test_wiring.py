"""The gate must be wired so that a failure costs nothing.

The dangerous ordering is `clear_database()` before validation: a parser regression would
then wipe a good graph and fail halfway through the reload, leaving nothing behind.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from legal_assistant import config
from legal_assistant.graph.loader import GraphLoader
from legal_assistant.pipelines.graph_build import build_graph
from legal_assistant.validation.gate import GraphValidationError

CELEX = "32022R0868"  # the smallest act, so the test stays quick


def _config():
    return {
        "html_file": str(config.CORPUS_DIR / f"{CELEX}.html"),
        "celex": CELEX,
        "eurolex_url": "",
        "document_info_url": "",
    }


@pytest.fixture(autouse=True)
def no_network():
    """The only part of the act pipeline that reaches out is the case-law metadata."""
    with patch(
        "legal_assistant.scraper.eurlex_exporter.EURLexHTMLParser._get_case_law",
        return_value=[],
    ):
        yield


def test_plan_document_writes_nothing():
    """Planning is pure: it must not touch the client it was constructed with."""
    graph = MagicMock()
    GraphLoader(graph).plan_document(_config())
    graph.upsert_graph_node.assert_not_called()
    graph.create_relationship.assert_not_called()


def test_load_document_writes_after_validating():
    graph = MagicMock()
    plan = GraphLoader(graph).load_document(_config())
    assert graph.upsert_graph_node.call_count == len(plan.node_ops)


def test_a_failing_act_stops_the_whole_batch():
    """One bad document must not leave the others half-written."""
    graph = MagicMock()
    loader = GraphLoader(graph)

    with patch.object(
        GraphLoader, "plan_document",
        side_effect=GraphValidationError("act BAD", []),
    ):
        with pytest.raises(RuntimeError, match="failed validation"):
            loader.load_all_documents([_config(), _config()])

    graph.upsert_graph_node.assert_not_called()


def test_database_is_not_cleared_when_validation_fails():
    """The ordering guarantee: validate everything, *then* clear, *then* write."""
    graph = MagicMock()

    with patch("legal_assistant.pipelines.graph_build.make_graph_client", return_value=graph), \
         patch("legal_assistant.pipelines.graph_build.EurlexDocumentUtils") as utils, \
         patch.object(GraphLoader, "plan_all_documents",
                      side_effect=RuntimeError("1/1 document(s) failed validation")):
        utils.return_value.build_document_config.return_value = _config()

        with pytest.raises(RuntimeError, match="failed validation"):
            build_graph([CELEX], clear_db=True)

    graph.clear_database.assert_not_called()
    graph.upsert_graph_node.assert_not_called()
    graph.close.assert_called_once()


def test_clear_happens_before_the_writes_when_validation_passes():
    graph = MagicMock()
    order = []
    graph.clear_database.side_effect = lambda: order.append("clear")
    graph.upsert_graph_node.side_effect = lambda **kw: order.append("write") or kw["node_properties"]["id"]
    graph.generate_text_embeddings.return_value = 1536

    with patch("legal_assistant.pipelines.graph_build.make_graph_client", return_value=graph), \
         patch("legal_assistant.pipelines.graph_build.EurlexDocumentUtils") as utils, \
         patch("legal_assistant.pipelines.graph_build.make_embeddings"):
        utils.return_value.build_document_config.return_value = _config()
        build_graph([CELEX], clear_db=True)

    assert order[0] == "clear", "the database must be cleared before the first write"
    assert order.count("clear") == 1
    assert "write" in order
