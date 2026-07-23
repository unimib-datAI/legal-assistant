"""Map analysed obligations onto graph nodes and edges.

Each AnalysedObligation becomes one Obligation node hanging off its source passage, with its
addressees and beneficiaries resolved to Actor edges through anchoring. The element lists are
reduced to the node's single fields, and the node's trust is the weakest method among them.
The obligation validation checks must pass on what is built.
"""
from __future__ import annotations

from collections import Counter

from legal_assistant.graph.recorder import RecordingGraph
from legal_assistant.obligations.anchoring import AnchorResult
from legal_assistant.obligations.builder import build_obligations
from legal_assistant.obligations.models import (
    AnalysedObligation,
    Actor,
    ExtractedElement,
    ExtractedPredicate,
    ExtractionMethod,
    ObligationType,
)
from legal_assistant.validation.checks import dangling_edges
from legal_assistant.validation.obligation_checks import (
    actor_targets_exist,
    obligations_anchored,
)
from legal_assistant.validation.plan import GraphPlan

AI_ACT = "32024R1689"
SOURCE = f"{AI_ACT}_016.0"
PROVIDER = Actor(id="provider", label="Provider", celex=AI_ACT)


def _obligation(**over) -> AnalysedObligation:
    fields = dict(
        id=f"{SOURCE}#ob_1",
        celex=AI_ACT,
        obligation_type=ObligationType.ACTION,
        predicate=ExtractedPredicate(value="shall ensure", verb="active",
                                     method=ExtractionMethod.STATED),
        addressees=[ExtractedElement(value="Providers of high-risk AI systems",
                                     method=ExtractionMethod.STATED)],
        beneficiaries=[ExtractedElement(value="End-users",
                                        method=ExtractionMethod.BACKGROUND)],
    )
    fields.update(over)
    return AnalysedObligation(**fields)


def _build(obligations, anchor, actors=(PROVIDER,)):
    graph = RecordingGraph()
    build_obligations(graph, list(obligations), anchor, list(actors))
    return GraphPlan.from_recorder(graph)


def _nodes(plan, label):
    return [n for n in plan.node_ops if n.label == label]


def _edges(plan, rel):
    return [e for e in plan.edge_ops if e.rel_type == rel]


def test_an_obligation_becomes_a_node_off_its_passage():
    anchor = AnchorResult(resolved={"Providers of high-risk AI systems": ["provider"]})
    plan = _build([_obligation(beneficiaries=[])], anchor)

    obligations = _nodes(plan, "Obligation")
    assert [o.id for o in obligations] == [f"{SOURCE}#ob_1"]
    states = _edges(plan, "STATES")
    assert (states[0].left_id, states[0].left_label, states[0].right_id) == (
        SOURCE, "Paragraph", f"{SOURCE}#ob_1",
    )


def test_an_annex_obligation_hangs_off_an_annex_point():
    annex_source = f"{AI_ACT}anx_IV.p_001"
    anchor = AnchorResult(resolved={"Providers of high-risk AI systems": ["provider"]})
    plan = _build([_obligation(id=f"{annex_source}#ob_1", beneficiaries=[])], anchor)

    states = _edges(plan, "STATES")
    assert states[0].left_label == "AnnexPoint"


def test_the_addressee_becomes_an_edge_to_the_resolved_actor():
    anchor = AnchorResult(resolved={"Providers of high-risk AI systems": ["provider"]})
    plan = _build([_obligation(beneficiaries=[])], anchor)

    addressed = _edges(plan, "ADDRESSED_TO")
    assert [(e.left_id, e.right_id) for e in addressed] == [(f"{SOURCE}#ob_1", "provider")]


def test_a_beneficiary_becomes_an_edge():
    anchor = AnchorResult(resolved={
        "Providers of high-risk AI systems": ["provider"],
        "End-users": ["end_users"],
    })
    plan = _build([_obligation()], anchor,
                  actors=(PROVIDER, Actor(id="end_users", label="End-users")))

    benefits = _edges(plan, "BENEFITS")
    assert [(e.left_id, e.right_id) for e in benefits] == [(f"{SOURCE}#ob_1", "end_users")]


def test_promoted_actors_are_written_with_their_hierarchy():
    child = Actor(id="providers_of_high_risk_ai_systems",
                  label="Providers of high-risk AI systems", celex=AI_ACT, is_a=["provider"])
    anchor = AnchorResult(
        resolved={"Providers of high-risk AI systems": ["providers_of_high_risk_ai_systems"]},
        promoted=[child],
    )
    plan = _build([_obligation(beneficiaries=[])], anchor)

    actor_ids = {n.id for n in _nodes(plan, "Actor")}
    assert "providers_of_high_risk_ai_systems" in actor_ids
    is_a = _edges(plan, "IS_A")
    assert (is_a[0].left_id, is_a[0].right_id) == ("providers_of_high_risk_ai_systems", "provider")


def test_the_nodes_weakest_method_is_the_least_reliable_element():
    """A background-inferred beneficiary drags the whole obligation's trust down."""
    anchor = AnchorResult(resolved={
        "Providers of high-risk AI systems": ["provider"], "End-users": ["end_users"],
    })
    plan = _build([_obligation()], anchor,
                  actors=(PROVIDER, Actor(id="end_users", label="End-users")))

    node = _nodes(plan, "Obligation")[0]
    assert node.properties["weakest_method"] == "BACKGROUND"


def test_the_vocabulary_actor_is_written_so_edges_have_a_target():
    """An ADDRESSED_TO edge to an actor with no node is silently dropped by Neo4j's MERGE.

    The graph never had the vocabulary loaded as nodes, only the actors.yaml file, so the
    build must write the actors it resolves onto or every addressee edge vanishes on replay.
    """
    anchor = AnchorResult(resolved={"Providers of high-risk AI systems": ["provider"]})
    plan = _build([_obligation(beneficiaries=[])], anchor)

    actor_ids = {n.id for n in _nodes(plan, "Actor")}
    assert "provider" in actor_ids


def test_what_is_built_has_no_dangling_edges():
    """Every edge endpoint exists in the same plan, so nothing is dropped on replay."""
    anchor = AnchorResult(resolved={
        "Providers of high-risk AI systems": ["provider"], "End-users": ["end_users"],
    })
    plan = _build([_obligation()], anchor,
                  actors=(PROVIDER, Actor(id="end_users", label="End-users")))
    assert dangling_edges(plan) == []


def test_what_is_built_passes_the_obligation_checks():
    anchor = AnchorResult(resolved={"Providers of high-risk AI systems": ["provider"]})
    plan = _build([_obligation(beneficiaries=[])], anchor)

    assert obligations_anchored(plan, {SOURCE}) == []
    assert actor_targets_exist(plan, [PROVIDER]) == []
