"""Fetch the text of cited provisions for the analysis prompt.

Our adaptation of the source repository's ``solve_references``. Theirs resolved paragraph ids
within an in-memory dataset; ours resolves article ids against the graph, since detection
records same-act article references. The analysis stage reads this so an obligation dispersed
across a cross-reference can still be reconstructed.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from legal_assistant.graph.queries import NodeQueries
from legal_assistant.obligations.references import citation_contents

AI_ACT = "32024R1689"


def _graph(rows):
    graph = MagicMock()

    def query(statement, params=None):
        if statement == NodeQueries.GET_ARTICLES_BY_IDS:
            return [r for r in rows if r["id"] in params["ids"]]
        return []

    graph.query.side_effect = query
    return graph


ROWS = [
    {"id": f"{AI_ACT}art_11", "title": "Technical documentation", "text": "The documentation ..."},
    {"id": f"{AI_ACT}art_18", "title": "Documentation keeping", "text": "The provider shall keep ..."},
]


def test_no_references_fetch_nothing():
    assert citation_contents(_graph(ROWS), []) == []


def test_a_cited_article_returns_its_titled_text():
    contents = citation_contents(_graph(ROWS), [f"{AI_ACT}art_11"])
    assert len(contents) == 1
    assert "Technical documentation" in contents[0]
    assert "The documentation ..." in contents[0]


def test_several_references_return_in_request_order():
    contents = citation_contents(_graph(ROWS), [f"{AI_ACT}art_18", f"{AI_ACT}art_11"])
    assert contents[0].startswith("[AI Act, Article — Documentation keeping]")
    assert contents[1].startswith("[AI Act, Article — Technical documentation]")


def test_an_unknown_reference_is_skipped_not_faked():
    contents = citation_contents(_graph(ROWS), [f"{AI_ACT}art_999"])
    assert contents == []
