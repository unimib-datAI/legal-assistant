"""Anchor extracted addressee strings onto the actor vocabulary.

Four outcomes, only the last discards: match an actor by meaning; else contain one and be
promoted as its qualified child; else recur often enough to be promoted alone; else land in
the unmapped report. Embeddings are faked so the similarity is under the test's control.
"""
from __future__ import annotations

from collections import Counter

import pytest

from legal_assistant.obligations.anchoring import anchor_strings, split_conjunctions
from legal_assistant.obligations.models import Actor

PROVIDER = Actor(id="provider", label="Provider", celex="32024R1689")
DEPLOYER = Actor(id="deployer", label="Deployer", celex="32024R1689")
VOCAB = [PROVIDER, DEPLOYER]

# A fake embedding: each phrase maps to a unit vector over a small concept space, so cosine
# similarity is 1 for phrases sharing a concept and 0 otherwise. Anything unknown is orthogonal.
_CONCEPTS = {
    "provider": (1.0, 0.0, 0.0),
    "the provider": (1.0, 0.0, 0.0),
    "deployer": (0.0, 1.0, 0.0),
    "the deployer": (0.0, 1.0, 0.0),
}


def _embed(texts):
    return [_CONCEPTS.get(t.strip().lower(), (0.0, 0.0, 1.0)) for t in texts]


def _anchor(strings, **kw):
    kw.setdefault("threshold", 0.9)
    kw.setdefault("frequency_floor", 3)
    return anchor_strings(Counter(strings), VOCAB, _embed, **kw)


# ── conjunction splitting ────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("text", "parts"),
    [
        ("The provider or its legal representative", ["The provider", "its legal representative"]),
        ("providers and deployers", ["providers", "deployers"]),
        ("the provider, the importer and the distributor",
         ["the provider", "the importer", "the distributor"]),
        ("the provider", ["the provider"]),
    ],
)
def test_split_conjunctions(text, parts):
    assert split_conjunctions(text) == parts


# ── outcome 1: match ─────────────────────────────────────────────────────────

def test_a_string_matching_an_actor_is_anchored_to_it():
    result = _anchor(["the provider"])
    assert result.resolved["the provider"] == ["provider"]
    assert result.promoted == []


def test_a_conjunction_anchors_to_several_actors():
    result = _anchor(["the provider and the deployer"])
    assert result.resolved["the provider and the deployer"] == ["provider", "deployer"]


# ── outcome 2: containment promotion ─────────────────────────────────────────

def test_a_qualified_string_is_promoted_as_a_child():
    """"providers of high-risk AI systems" contains "provider" but does not match it."""
    result = _anchor(["providers of high-risk AI systems"])

    promoted = {a.id: a for a in result.promoted}
    assert "providers_of_high_risk_ai_systems" in promoted
    assert promoted["providers_of_high_risk_ai_systems"].is_a == ["provider"]
    assert result.resolved["providers of high-risk AI systems"] == [
        "providers_of_high_risk_ai_systems"
    ]


# ── outcome 3: frequency promotion ───────────────────────────────────────────

def test_a_recurring_unknown_string_is_promoted_alone():
    """"Member State" matches and contains no known actor, but recurs, so it becomes an actor."""
    result = _anchor(["Member State"] * 3)

    promoted = {a.id: a for a in result.promoted}
    assert "member_state" in promoted
    assert promoted["member_state"].is_a == []
    assert result.resolved["Member State"] == ["member_state"]


def test_a_one_off_unknown_string_is_not_promoted():
    result = _anchor(["Member State"])
    assert result.promoted == []
    assert "Member State" in result.unmapped


# ── outcome 4: unmapped ──────────────────────────────────────────────────────

def test_an_unmatched_one_off_lands_in_the_unmapped_report_with_its_count():
    result = _anchor(["a one-off phrase", "a one-off phrase"], frequency_floor=3)
    assert result.unmapped["a one-off phrase"] == 2
    assert "a one-off phrase" not in result.resolved


def test_promotion_is_idempotent_against_the_existing_vocabulary():
    """A qualified form already in the vocabulary is not promoted a second time."""
    existing = Actor(id="providers_of_high_risk_ai_systems",
                     label="Providers of high-risk AI systems", celex="32024R1689",
                     is_a=["provider"])
    result = anchor_strings(
        Counter(["providers of high-risk AI systems"]),
        VOCAB + [existing], _embed, threshold=0.9, frequency_floor=3,
    )
    assert result.promoted == []
    assert result.resolved["providers of high-risk AI systems"] == [
        "providers_of_high_risk_ai_systems"
    ]
