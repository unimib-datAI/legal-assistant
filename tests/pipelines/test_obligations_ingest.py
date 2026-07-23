"""The obligations ingestion flow: detect, filter, analyse, anchor, build, validate.

The LLM stages are injected so the orchestration is testable without a model: a fake extract
returns canned analysed obligations, and the test checks that their addressee strings are
anchored and that the built subgraph carries the expected nodes and edges.
"""
from __future__ import annotations

from legal_assistant.graph.recorder import RecordingGraph
from legal_assistant.obligations.models import (
    Actor,
    AnalysedObligation,
    ExtractedElement,
    ExtractedPredicate,
    ExtractionMethod,
    ObligationType,
)
from unittest.mock import MagicMock

from legal_assistant.graph.queries import NodeQueries
from legal_assistant.pipelines.obligations_ingest import (
    build_plan_from_obligations,
    extract_obligations,
)

AI_ACT = "32024R1689"
SOURCE = f"{AI_ACT}_016.0"
PROVIDER = Actor(id="provider", label="Provider", celex=AI_ACT)


def _embed(texts):
    # "providers of high-risk AI systems" contains "provider" -> promoted, no vector hit needed.
    concepts = {"provider": (1.0, 0.0)}
    return [concepts.get(t.strip().lower(), (0.0, 1.0)) for t in texts]


def _obligation(source=SOURCE):
    return AnalysedObligation(
        id=f"{source}#ob_1",
        celex=AI_ACT,
        obligation_type=ObligationType.ACTION,
        predicate=ExtractedPredicate(value="shall ensure", method=ExtractionMethod.STATED),
        addressees=[ExtractedElement(value="providers of high-risk AI systems",
                                     method=ExtractionMethod.STATED)],
    )


def _build(obligations, passage_ids):
    return build_plan_from_obligations(
        obligations, [PROVIDER], passage_ids, _embed,
        threshold=0.9, frequency_floor=3,
    )


def test_the_addressee_string_is_promoted_and_edged():
    plan, anchor = _build([_obligation()], {SOURCE})

    actor_ids = {n.id for n in plan.node_ops if n.label == "Actor"}
    assert "providers_of_high_risk_ai_systems" in actor_ids

    addressed = [e for e in plan.edge_ops if e.rel_type == "ADDRESSED_TO"]
    assert [(e.left_id, e.right_id) for e in addressed] == [
        (f"{SOURCE}#ob_1", "providers_of_high_risk_ai_systems")
    ]


def test_the_built_plan_passes_the_obligation_checks():
    """build_plan_from_obligations validates internally, so a good input raises nothing."""
    plan, _ = _build([_obligation()], {SOURCE})
    assert any(n.label == "Obligation" for n in plan.node_ops)


def test_two_deontic_sentences_in_one_passage_get_distinct_ids():
    """A paragraph with two duties must not give both obligations the same id.

    Each sentence is analysed on its own and numbered from 1, so without a passage-level
    counter the second obligation reuses #ob_1 and overwrites the first in the graph.
    """
    graph = MagicMock()

    def query(statement, params=None):
        if statement == NodeQueries.GET_PARAGRAPHS_BY_ACTS:
            return [{"id": SOURCE, "celex": AI_ACT,
                     "text": "The provider shall do X. The provider shall do Y."}]
        return []

    graph.query.side_effect = query

    analysis = (
        "```json\n[{"
        '"ObligationTypeClassification": "Obligation of Action",'
        '"Predicate": {"value": "shall do", "verb": "active", "extraction_method": "Stated"}'
        "}]\n```"
    )
    filtering = '```json\n{"classification": "Deontic obligation", "justification": "d"}\n```'

    llm = MagicMock()
    # The analysis system prompt is the only one titled "... Syntactical Analysis ...".
    llm.invoke.side_effect = lambda messages: type(
        "Msg", (), {"content": analysis if "syntactical" in messages[0].content.lower()
                    else filtering}
    )()

    obligations = extract_obligations(graph, llm, [AI_ACT])

    ids = [o.id for o in obligations]
    assert len(ids) == 2, f"expected two obligations, got {ids}"
    assert len(ids) == len(set(ids)), f"duplicate obligation ids: {ids}"


def test_passages_are_processed_concurrently_without_id_collision():
    """Two passages run in parallel; each keeps its own ordinal, and order is preserved."""
    graph = MagicMock()
    passages = [f"{AI_ACT}_016.0", f"{AI_ACT}_017.0"]

    def query(statement, params=None):
        if statement == NodeQueries.GET_PARAGRAPHS_BY_ACTS:
            return [{"id": pid, "celex": AI_ACT, "text": "The provider shall do X."}
                    for pid in passages]
        return []

    graph.query.side_effect = query

    analysis = (
        "```json\n[{"
        '"ObligationTypeClassification": "Obligation of Action",'
        '"Predicate": {"value": "shall do", "verb": "active", "extraction_method": "Stated"}'
        "}]\n```"
    )
    filtering = '```json\n{"classification": "Deontic obligation", "justification": "d"}\n```'
    llm = MagicMock()
    llm.invoke.side_effect = lambda messages: type(
        "Msg", (), {"content": analysis if "syntactical" in messages[0].content.lower()
                    else filtering}
    )()

    obligations = extract_obligations(graph, llm, [AI_ACT], max_workers=4)

    ids = [o.id for o in obligations]
    assert ids == [f"{AI_ACT}_016.0#ob_1", f"{AI_ACT}_017.0#ob_1"]


def test_reset_deletes_only_the_targeted_acts_obligations():
    graph = MagicMock()
    from legal_assistant.pipelines.obligations_ingest import delete_obligations

    delete_obligations(graph, ["32016R0679"])

    statement, params = graph.query.call_args.args[0], graph.query.call_args.kwargs.get("params") \
        or graph.query.call_args.args[1]
    assert statement == NodeQueries.DELETE_OBLIGATIONS_BY_ACTS
    assert params == {"acts": ["32016R0679"]}


def test_an_obligation_off_an_unknown_passage_is_rejected():
    """The anchor is checked: an obligation whose passage is not in the graph must fail."""
    import pytest

    from legal_assistant.validation.gate import GraphValidationError
    with pytest.raises(GraphValidationError):
        _build([_obligation(source=f"{AI_ACT}_999.001")], {SOURCE})
