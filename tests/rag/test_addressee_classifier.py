"""The separate addressee classifier for the obligations branch.

Kept apart from the shared query classifier: it scores the roles a question is about, so the
obligations branch can filter by them, without changing the prompt every RAG query uses.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from legal_assistant.graph.queries import NodeQueries
from legal_assistant.rag.intent_classifier import ActorRelevance, QueryClassifier, RawAddressees


def _classifier(scored):
    graph = MagicMock()
    graph.query.return_value = [
        {"id": "provider", "label": "Provider"},
        {"id": "deployer", "label": "Deployer"},
        {"id": "controller", "label": "Controller"},
    ]
    llm = MagicMock()
    # with_structured_output(RawAddressees) -> object whose .invoke returns a RawAddressees
    llm.with_structured_output.return_value.invoke.return_value = RawAddressees(
        addressee_relevances=[ActorRelevance(actor=a, relevance=s) for a, s in scored]
    )
    return QueryClassifier(graph, llm, addressee_score_threshold=0.5)


def test_a_role_above_threshold_is_selected():
    clf = _classifier([("provider", 0.9), ("deployer", 0.1), ("controller", 0.0)])
    addressees, scores = clf.classify_addressees("what must a provider do")
    assert addressees == ["provider"]
    assert scores["provider"] == 0.9


def test_roles_are_ordered_by_score():
    clf = _classifier([("provider", 0.6), ("deployer", 0.9), ("controller", 0.0)])
    addressees, _ = clf.classify_addressees("duties of deployers and providers")
    assert addressees == ["deployer", "provider"]


def test_no_role_above_threshold_returns_empty():
    clf = _classifier([("provider", 0.2), ("deployer", 0.1), ("controller", 0.0)])
    addressees, _ = clf.classify_addressees("what is personal data")
    assert addressees == []


def test_the_actor_list_reaches_the_prompt():
    clf = _classifier([("provider", 0.9)])
    clf.classify_addressees("q")
    prompt = clf._structured_addressees.invoke.call_args.args[0]
    assert "provider" in prompt and "Provider" in prompt
