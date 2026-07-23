"""Checks that must fail before an obligation reaches Neo4j.

Same principle as the act gate: a wrong obligation is worse than a missing one, because it
reads as a duty someone owes. Each check is proved on a plan crafted to break it, not only on
a good one, since a check that has never failed is a check nobody knows works.
"""
from __future__ import annotations

import pytest

from legal_assistant.graph.recorder import RecordingGraph
from legal_assistant.obligations.models import Actor
from legal_assistant.validation.obligation_checks import (
    actor_targets_exist,
    hierarchy_terminates,
    obligations_anchored,
    unmapped_within_ceiling,
)
from legal_assistant.validation.plan import GraphPlan

PASSAGES = {"32024R1689_023.003", "32024R1689anx_IV.p_007"}


def _plan(build) -> GraphPlan:
    recorder = RecordingGraph()
    build(recorder)
    return GraphPlan.from_recorder(recorder)


def _obligation_node(graph, obligation_id):
    graph.upsert_graph_node("Obligation", {"id": obligation_id, "celex": "32024R1689"})


# ── anchoring ────────────────────────────────────────────────────────────────

def test_an_obligation_anchored_to_a_real_passage_passes():
    plan = _plan(lambda g: _obligation_node(g, "32024R1689_023.003#ob_1"))
    assert obligations_anchored(plan, PASSAGES) == []


def test_an_obligation_anchored_to_an_annex_point_passes():
    plan = _plan(lambda g: _obligation_node(g, "32024R1689anx_IV.p_007#ob_1"))
    assert obligations_anchored(plan, PASSAGES) == []


def test_an_obligation_on_a_passage_that_does_not_exist_is_caught():
    """The failure mode of joining on the source pipeline's own paragraph ids."""
    plan = _plan(lambda g: _obligation_node(g, "32024R1689_999.001#ob_1"))
    violations = obligations_anchored(plan, PASSAGES)
    assert [v.kind for v in violations] == ["obligation_unanchored"]


def test_an_obligation_whose_id_carries_no_marker_is_caught():
    plan = _plan(lambda g: _obligation_node(g, "32024R1689_023.003"))
    assert [v.kind for v in obligations_anchored(plan, PASSAGES)] == ["obligation_bad_id"]


# ── hierarchy ────────────────────────────────────────────────────────────────

def test_a_well_formed_hierarchy_passes():
    actors = [
        Actor(id="provider", label="Provider"),
        Actor(id="provider_of_high_risk", label="Provider of a high-risk AI system",
              is_a=["provider"]),
    ]
    assert hierarchy_terminates(actors) == []


def test_a_cycle_is_caught_and_does_not_hang():
    """The filter walks IS_A*0.., so a cycle is an unbounded traversal at query time."""
    actors = [
        Actor(id="a", label="A", is_a=["b"]),
        Actor(id="b", label="B", is_a=["a"]),
    ]
    assert {v.kind for v in hierarchy_terminates(actors)} == {"actor_cycle"}


def test_an_actor_that_qualifies_itself_is_caught():
    actors = [Actor(id="a", label="A", is_a=["a"])]
    assert {v.kind for v in hierarchy_terminates(actors)} == {"actor_cycle"}


def test_qualifying_an_actor_that_does_not_exist_is_caught():
    """A parent nobody defined means the filter stops short of the duties it should reach."""
    actors = [Actor(id="a", label="A", is_a=["ghost"])]
    assert [v.kind for v in hierarchy_terminates(actors)] == ["actor_unknown_parent"]


# ── edge targets ─────────────────────────────────────────────────────────────

def test_addressing_an_actor_outside_the_vocabulary_is_caught():
    def build(graph):
        _obligation_node(graph, "32024R1689_023.003#ob_1")
        graph.create_relationship(
            "Obligation", "Actor", "32024R1689_023.003#ob_1", "ghost", "ADDRESSED_TO")

    violations = actor_targets_exist(_plan(build), [Actor(id="provider", label="Provider")])
    assert [v.kind for v in violations] == ["actor_not_in_vocabulary"]


def test_a_beneficiary_outside_the_vocabulary_is_caught():
    def build(graph):
        _obligation_node(graph, "32024R1689_023.003#ob_1")
        graph.create_relationship(
            "Obligation", "Actor", "32024R1689_023.003#ob_1", "ghost", "BENEFITS")

    violations = actor_targets_exist(_plan(build), [Actor(id="provider", label="Provider")])
    assert [v.kind for v in violations] == ["actor_not_in_vocabulary"]


# ── unmapped ceiling ─────────────────────────────────────────────────────────

def test_unmapped_below_the_ceiling_passes():
    assert unmapped_within_ceiling(unmapped=5, total=100, ceiling=0.1) == []


def test_unmapped_above_the_ceiling_is_caught():
    """A high unmapped share means the generation missed a subject the acts define."""
    violations = unmapped_within_ceiling(unmapped=20, total=100, ceiling=0.1)
    assert [v.kind for v in violations] == ["addressees_unmapped"]


def test_nothing_extracted_is_not_a_ceiling_breach():
    """Zero of zero is not 100%: an empty run is a different problem, reported elsewhere."""
    assert unmapped_within_ceiling(unmapped=0, total=0, ceiling=0.1) == []
