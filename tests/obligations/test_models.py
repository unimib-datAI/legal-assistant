"""The graph-node shapes for obligations and the actors they are addressed to.

The extraction record itself is deliberately not modelled yet: phase 3 pins the source
pipeline's real output as a fixture first, so that the reader is written against what the
code emits rather than against the paper's listings.
"""
from __future__ import annotations

import pytest

from legal_assistant.obligations.models import (
    Actor,
    ExtractionMethod,
    Modality,
    Obligation,
    ObligationType,
    source_id_of,
)


def _obligation(**overrides) -> Obligation:
    """A minimal well-formed obligation: only the predicate is mandatory in the framework."""
    fields = dict(
        id="32024R1689_023.003#ob_1",
        celex="32024R1689",
        modality=Modality.OBLIGATION,
        obligation_type=ObligationType.ACTION,
        predicate_text="shall indicate their name",
        predicate_method=ExtractionMethod.STATED,
    )
    fields.update(overrides)
    return Obligation(**fields)


def test_source_id_is_recoverable_from_the_obligation_id():
    """The retrieval bridge reads the source passage off the id, with no round trip."""
    assert source_id_of("32024R1689_023.003#ob_1") == "32024R1689_023.003"


def test_source_id_of_an_annex_obligation():
    """Annex points are passages too, and their ids carry dots of their own."""
    assert source_id_of("32024R1689anx_IV.p_007#ob_2") == "32024R1689anx_IV.p_007"


def test_source_id_rejects_an_id_with_no_marker():
    with pytest.raises(ValueError):
        source_id_of("32024R1689_023.003")


def test_weakest_method_is_the_least_reliable_populated_element():
    """STATED > CONTEXT > CITATION > BACKGROUND: the answer is only as good as its weakest part."""
    obligation = _obligation(
        addressee_method=ExtractionMethod.STATED,
        target_method=ExtractionMethod.CONTEXT,
    )
    assert obligation.weakest_method == ExtractionMethod.CONTEXT


def test_background_knowledge_dominates_the_weakest_method():
    """An element the model inferred from nothing in the text governs how far the whole is trusted."""
    obligation = _obligation(
        addressee_method=ExtractionMethod.BACKGROUND,
        target_method=ExtractionMethod.CONTEXT,
        specification_method=ExtractionMethod.CITATION,
    )
    assert obligation.weakest_method == ExtractionMethod.BACKGROUND


def test_absent_elements_do_not_weaken_the_obligation():
    """NONE marks an element that is not there, which is not the same as one badly extracted."""
    obligation = _obligation(
        addressee_method=ExtractionMethod.NONE,
        target_method=ExtractionMethod.NONE,
        beneficiary_method=ExtractionMethod.NONE,
    )
    assert obligation.weakest_method == ExtractionMethod.STATED


def test_an_obligation_with_nothing_populated_reports_none():
    """Degenerate, but it must not raise: NONE is a value, not a missing case."""
    obligation = _obligation(predicate_method=ExtractionMethod.NONE)
    assert obligation.weakest_method == ExtractionMethod.NONE


def test_an_actor_may_qualify_another():
    """"Provider of a high-risk AI system" is a provider under a qualification, not a peer."""
    actor = Actor(
        id="provider_of_high_risk_ai_system",
        label="Provider of a high-risk AI system",
        celex="32024R1689",
        is_a=["provider"],
    )
    assert actor.is_a == ["provider"]


def test_a_cross_cutting_actor_belongs_to_no_single_act():
    """The Commission bears duties under several acts, so it is not owned by one."""
    actor = Actor(id="european_commission", label="European Commission")
    assert actor.celex is None
    assert actor.is_a == []
