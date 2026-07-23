"""The compliance checklist: every obligation an actor bears, rendered by the LLM.

Deterministic by construction. The query returns the complete set for (act, actor), hierarchy
included, ordered by provision; the LLM only renders it, so nothing is silently truncated the
way a top-k retrieval would truncate it.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from legal_assistant.graph.queries import NodeQueries
from legal_assistant.pipelines.obligation_checklist import checklist, fetch_checklist

AI_ACT = "32024R1689"

ROWS = [
    {"id": f"{AI_ACT}_016.0#ob_1", "source_id": f"{AI_ACT}_016.0", "actor": "provider",
     "modality": "OBLIGATION", "obligation_type": "ACTION", "predicate_text": "shall ensure",
     "target": "compliance", "specification": None, "precondition": None,
     "weakest_method": "STATED"},
    {"id": f"{AI_ACT}_016.0#ob_2", "source_id": f"{AI_ACT}_016.0",
     "actor": "provider_of_high_risk_ai_system", "modality": "OBLIGATION",
     "obligation_type": "ACTION", "predicate_text": "shall keep the documentation",
     "target": None, "specification": None, "precondition": None, "weakest_method": "STATED"},
]


def _graph(rows):
    graph = MagicMock()
    graph.query.return_value = rows
    return graph


def test_fetch_uses_the_actor_query_with_bound_params():
    graph = _graph(ROWS)
    fetch_checklist(graph, AI_ACT, "provider")

    statement = graph.query.call_args.args[0]
    params = graph.query.call_args.kwargs.get("params") or graph.query.call_args.args[1]
    assert statement == NodeQueries.GET_OBLIGATIONS_FOR_ACTOR
    assert params == {"celex": AI_ACT, "actor": "provider"}


def test_fetch_returns_the_full_set_including_the_qualified_child():
    """A checklist for 'provider' must include the high-risk provider's obligation too."""
    rows = fetch_checklist(_graph(ROWS), AI_ACT, "provider")
    actors = {r["actor"] for r in rows}
    assert actors == {"provider", "provider_of_high_risk_ai_system"}


def test_checklist_renders_through_the_llm():
    llm = MagicMock()
    llm.invoke.return_value = type("Msg", (), {"content": "1. Ensure compliance ..."})()

    text = checklist(_graph(ROWS), llm, AI_ACT, "provider")

    assert text == "1. Ensure compliance ..."
    prompt = " ".join(m.content for m in llm.invoke.call_args.args[0])
    assert "shall ensure" in prompt
    assert "shall keep the documentation" in prompt


def test_an_actor_with_no_obligations_yields_an_empty_checklist_not_an_error():
    llm = MagicMock()
    text = checklist(_graph([]), llm, AI_ACT, "nobody")
    assert text == ""
    llm.invoke.assert_not_called()
