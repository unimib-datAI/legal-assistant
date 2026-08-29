"""Resume predicates: is this unit already in the graph, whole and validated?

Both pipelines ask before fetching anything, so a wrong answer is expensive in both
directions: a false "done" silently leaves a gap in the corpus, a false "not done" re-runs
work that Playwright and Cellar already paid for.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from legal_assistant.graph.completion import act_is_loaded, case_law_is_ingested


def _graph(rows: list[dict]) -> MagicMock:
    graph = MagicMock()
    graph.query.return_value = rows
    return graph


@pytest.mark.parametrize("predicate", [act_is_loaded, case_law_is_ingested])
def test_true_when_the_graph_says_done(predicate):
    assert predicate(_graph([{"done": True}]), "32016R0679") is True


@pytest.mark.parametrize("predicate", [act_is_loaded, case_law_is_ingested])
def test_false_when_the_graph_says_not_done(predicate):
    assert predicate(_graph([{"done": False}]), "32016R0679") is False


@pytest.mark.parametrize("predicate", [act_is_loaded, case_law_is_ingested])
def test_no_rows_reads_as_not_done(predicate):
    """A judgment with no stub at all returns nothing. That is "not done", not a crash."""
    assert predicate(_graph([]), "62019CJ0645") is False


@pytest.mark.parametrize("predicate", [act_is_loaded, case_law_is_ingested])
def test_the_celex_is_a_parameter_never_interpolated(predicate):
    """Cypher is parameterized everywhere in this codebase; these are no exception."""
    graph = _graph([{"done": True}])
    predicate(graph, "32016R0679")

    query, params = graph.query.call_args[0]
    assert params == {"celex": "32016R0679"}
    assert "32016R0679" not in query


def test_case_law_asks_for_sections_not_for_the_node():
    """The stub exists from the act build onward, so node presence would always say yes."""
    graph = _graph([{"done": False}])
    case_law_is_ingested(graph, "62019CJ0645")

    query = graph.query.call_args[0][0]
    assert "HAS_SECTION" in query
