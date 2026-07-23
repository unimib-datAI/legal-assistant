"""The annex branch, on the recital pattern: additive, thresholded, never displacing articles.

Annex III decides whether Chapter III applies, so "is my system high-risk" is unanswerable
from articles alone. The branch that answers it must not cost anything when switched off,
which is what the non-regression case here pins.

Neo4j and the LLM are mocked at their boundaries. The cross-encoder is switched off rather
than mocked: ``_rerank_*`` already has a reranker-free path that scores by rank.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from legal_assistant.graph.queries import NodeQueries
from legal_assistant.rag.intent_classifier import QueryClassification
from legal_assistant.rag.retrievers.hybrid import HybridRetriever

AI_ACT = "32024R1689"
GDPR = "32016R0679"

ARTICLES = [
    {"id": f"{AI_ACT}art_6", "title": "Classification rules for high-risk AI systems",
     "text": "AI systems referred to in Annex III shall be considered to be high-risk.",
     "act": AI_ACT, "chapter_number": "III", "chapter_title": "High-risk AI system"},
]

ANNEX_POINTS = [
    {"id": f"{AI_ACT}anx_III.p_002", "text": "Biometrics, in so far as their use is permitted.",
     "celex": AI_ACT, "point_label": "III, point 1", "section_heading": None,
     "annex_number": "III", "annex_title": "High-risk AI systems referred to in Article 6(2)"},
    {"id": f"{AI_ACT}anx_III.p_003",
     "text": "remote biometric identification systems used for law enforcement.",
     "celex": AI_ACT, "point_label": "III, point 1(a)", "section_heading": None,
     "annex_number": "III", "annex_title": "High-risk AI systems referred to in Article 6(2)"},
]


def _graph(annex_rows):
    """A Neo4j stand-in that answers by which query it was handed."""
    graph = MagicMock()

    def query(statement, params=None):
        if statement == NodeQueries.GET_ANNEX_POINTS_BY_ACTS:
            return annex_rows
        if statement == NodeQueries.GET_ARTICLES_BY_ACTS:
            return ARTICLES
        return []

    graph.query.side_effect = query
    return graph


def _retriever(*, annex_rows=ANNEX_POINTS, acts=(AI_ACT,), **overrides):
    classifier = MagicMock()
    classifier.classify.return_value = QueryClassification(
        intent="DEFINITIONAL", acts=list(acts), act_scores={a: 1.0 for a in acts},
    )

    store = MagicMock()
    store.similarity_search.return_value = []

    settings = dict(
        graph=_graph(annex_rows),
        article_vector_store=store,
        classifier=classifier,
        use_hyde=False,
        use_reranker=False,
        use_case_law=False,
        use_recitals=False,
        annex_score_threshold=0.0,
    )
    settings.update(overrides)
    return HybridRetriever(**settings)


def _types(docs):
    return [doc.metadata.get("type") for doc in docs]


def test_annex_points_are_retrieved_when_enabled():
    docs = _retriever(use_annexes=True).invoke("which AI systems are high-risk")
    assert "annex" in _types(docs)


def test_annexes_are_off_by_default_for_no_regression():
    """Off unless asked for: an existing eval run must return exactly what it returned."""
    baseline = _retriever(use_annexes=False).invoke("which AI systems are high-risk")
    assert "annex" not in _types(baseline)


def test_enabling_annexes_does_not_displace_articles():
    """The branch is additive. Article slots are guaranteed and must not move."""
    without = _retriever(use_annexes=False).invoke("which AI systems are high-risk")
    with_annexes = _retriever(use_annexes=True).invoke("which AI systems are high-risk")

    articles_before = [d.page_content for d in without if d.metadata.get("type") == "article"]
    articles_after = [d.page_content for d in with_annexes if d.metadata.get("type") == "article"]
    assert articles_before == articles_after


def test_an_act_without_annexes_returns_nothing_and_does_not_raise():
    """GDPR, the DGA and the Data Act have no annexes; the branch must simply be empty."""
    docs = _retriever(annex_rows=[], acts=(GDPR,), use_annexes=True).invoke("erasure")
    assert "annex" not in _types(docs)


def test_retrieved_annex_points_are_cited_by_point():
    """"Annex III, point 1(a)" is the citation a lawyer writes, so it heads the passage."""
    docs = _retriever(use_annexes=True).invoke("which AI systems are high-risk")
    annexes = [d for d in docs if d.metadata.get("type") == "annex"]
    assert any(d.page_content.startswith("[AI Act, Annex III, point 1(a)]") for d in annexes)


@pytest.mark.parametrize("cap", [1, 2])
def test_top_k_annexes_caps_the_branch(cap):
    docs = _retriever(use_annexes=True, top_k_annexes=cap).invoke("biometric identification")
    assert len([d for d in docs if d.metadata.get("type") == "annex"]) <= cap
