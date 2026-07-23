"""The obligations branch in HybridRetriever: additive, role-filtered, off by default.

Mirrors the annex and case-law branches. It filters obligations by the classified addressee
role (walking IS_A), reranks them, and adds the survivors in their own slots. Like the annex
branch it is off unless asked for, so an existing eval run is unchanged, and it degrades to
nothing when the classifier names no addressee.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from legal_assistant.graph.queries import NodeQueries
from legal_assistant.rag.intent_classifier import QueryClassification
from legal_assistant.rag.retrievers.hybrid import HybridRetriever

AI_ACT = "32024R1689"

ARTICLES = [
    {"id": f"{AI_ACT}art_16", "title": "Obligations of providers", "text": "Providers shall ...",
     "act": AI_ACT, "chapter_number": "III", "chapter_title": "High-risk AI system"},
]

OBLIGATIONS = [
    {"id": f"{AI_ACT}_016.0#ob_1", "source_id": f"{AI_ACT}_016.0", "actor": "provider",
     "modality": "OBLIGATION", "obligation_type": "ACTION",
     "predicate_text": "shall ensure their high-risk AI systems are compliant",
     "target": "the requirements of Section 2", "specification": None, "precondition": None,
     "weakest_method": "STATED"},
    {"id": f"{AI_ACT}_016.0#ob_2", "source_id": f"{AI_ACT}_016.0", "actor": "provider",
     "modality": "OBLIGATION", "obligation_type": "ACTION",
     "predicate_text": "shall keep the technical documentation", "target": None,
     "specification": None, "precondition": None, "weakest_method": "STATED"},
]


def _graph(obligation_rows):
    graph = MagicMock()

    def query(statement, params=None):
        if statement == NodeQueries.GET_OBLIGATIONS_FOR_ACTORS:
            return obligation_rows
        if statement == NodeQueries.GET_ARTICLES_BY_ACTS:
            return ARTICLES
        return []

    graph.query.side_effect = query
    return graph


def _retriever(*, addressees=("provider",), obligation_rows=OBLIGATIONS, **overrides):
    classifier = MagicMock()
    classifier.classify.return_value = QueryClassification(
        intent="DEFINITIONAL", acts=[AI_ACT], act_scores={AI_ACT: 1.0},
        addressees=list(addressees),
    )
    # The branch falls back to its own addressee classifier when the main one names none.
    classifier.classify_addressees.return_value = (list(addressees), {})
    store = MagicMock()
    store.similarity_search.return_value = []

    settings = dict(
        graph=_graph(obligation_rows), article_vector_store=store, classifier=classifier,
        use_hyde=False, use_reranker=False, use_case_law=False, use_recitals=False,
        obligation_score_threshold=0.0,
    )
    settings.update(overrides)
    return HybridRetriever(**settings)


def _types(docs):
    return [d.metadata.get("type") for d in docs]


def test_obligations_are_retrieved_for_the_classified_role():
    docs = _retriever(use_obligations=True).invoke("what must a provider do")
    assert "obligation" in _types(docs)


def test_obligations_are_off_by_default():
    docs = _retriever(use_obligations=False).invoke("what must a provider do")
    assert "obligation" not in _types(docs)


def test_no_classified_addressee_means_no_obligation_branch():
    """With no role named, the branch degrades to nothing rather than guessing."""
    docs = _retriever(use_obligations=True, addressees=()).invoke("what does Article 16 say")
    assert "obligation" not in _types(docs)


def test_enabling_obligations_does_not_displace_articles():
    without = _retriever(use_obligations=False).invoke("what must a provider do")
    with_ob = _retriever(use_obligations=True).invoke("what must a provider do")
    a_before = [d.page_content for d in without if d.metadata.get("type") == "article"]
    a_after = [d.page_content for d in with_ob if d.metadata.get("type") == "article"]
    assert a_before == a_after


def test_a_retrieved_obligation_renders_its_predicate_and_source():
    docs = _retriever(use_obligations=True).invoke("what must a provider do")
    obligations = [d for d in docs if d.metadata.get("type") == "obligation"]
    assert any("shall ensure their high-risk AI systems are compliant" in d.page_content
               for d in obligations)
    assert all(d.metadata.get("source_id") for d in obligations)
