"""The actor vocabulary is generated from the acts' own Definitions articles, then committed.

Generated, never hand-written, so adding an act costs nothing. Committed, so a regeneration
that quietly changes what "provider" means shows up as a diff instead of moving the meaning
of every role filter underneath the graph.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from legal_assistant.obligations.models import Actor
from legal_assistant.obligations.vocabulary import (
    DefinedSubject,
    DefinedSubjects,
    dumps,
    generate,
    loads,
    slugify,
)

AI_ACT = "32024R1689"

DEFINITION_ROWS = [
    {"id": f"{AI_ACT}_003.003", "text": "'provider' means a natural or legal person that develops an AI system"},
    {"id": f"{AI_ACT}_003.004", "text": "'deployer' means a natural or legal person using an AI system"},
]


def _llm(subjects):
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = DefinedSubjects(
        subjects=subjects
    )
    return llm


def _graph(rows=DEFINITION_ROWS):
    graph = MagicMock()
    graph.query.return_value = rows
    return graph


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Provider", "provider"),
        ("Provider of a high-risk AI system", "provider_of_a_high_risk_ai_system"),
        ("Market surveillance authority", "market_surveillance_authority"),
        ("  Data  holder  ", "data_holder"),
    ],
)
def test_slugify_makes_a_stable_id(label, expected):
    """Ids go into Cypher and into a committed file, so they must be plain and repeatable."""
    assert slugify(label) == expected


def test_generate_turns_defined_subjects_into_actors():
    actors = generate(
        _graph(),
        _llm([
            DefinedSubject(label="Provider", defined_in=f"{AI_ACT}_003.003"),
            DefinedSubject(label="Deployer", defined_in=f"{AI_ACT}_003.004"),
        ]),
        acts=[AI_ACT],
    )

    assert [(a.id, a.celex, a.defined_in) for a in actors] == [
        ("deployer", AI_ACT, f"{AI_ACT}_003.004"),
        ("provider", AI_ACT, f"{AI_ACT}_003.003"),
    ]


def test_aggregate_definitions_become_a_hierarchy():
    """The only hierarchy a Definitions article carries, and it runs upward.

    AI Act Article 3(8) reads "'operator' means a provider, product manufacturer, deployer,
    authorised representative, importer or distributor", which makes each of those an
    operator. Qualified forms such as "provider of a high-risk AI system" are defined nowhere
    in Article 3 and cannot come from this stage; they are promoted during anchoring, from
    the addressee strings the extraction produces.
    """
    actors = generate(
        _graph(),
        _llm([
            DefinedSubject(label="Operator"),
            DefinedSubject(label="Provider", is_a=["Operator"]),
            DefinedSubject(label="Deployer", is_a=["Operator"]),
        ]),
        acts=[AI_ACT],
    )

    assert {a.id: a.is_a for a in actors} == {
        "operator": [],
        "provider": ["operator"],
        "deployer": ["operator"],
    }


def test_a_defined_in_the_model_decorated_is_still_matched():
    """Models copy the id together with whatever punctuation labels it in the prompt.

    Against the live GDPR definitions this silently nulled every defined_in: the model
    returned '[32016R0679_004.7]', the guard against invented ids compared it to the real
    ids, and dropped all nine. The guard was right; the id just needed unwrapping first.
    """
    actors = generate(
        _graph(),
        _llm([DefinedSubject(label="Provider", defined_in=f"[{AI_ACT}_003.003]")]),
        acts=[AI_ACT],
    )

    assert actors[0].defined_in == f"{AI_ACT}_003.003"


def test_a_defined_in_outside_the_act_is_dropped():
    """The model may invent a citation; only ids the query actually returned are kept."""
    actors = generate(
        _graph(),
        _llm([DefinedSubject(label="Provider", defined_in="32016R0679_099.999")]),
        acts=[AI_ACT],
    )

    assert actors[0].defined_in is None


def test_a_subject_defined_by_two_acts_becomes_cross_cutting():
    """A Member State bears duties under every act, so it belongs to none of them.

    Keeping one actor per act would split those duties in two, and a filter on either half
    would return an answer that is incomplete without saying so.
    """
    actors = generate(_graph(), _llm([DefinedSubject(label="Member State")]),
                      acts=[AI_ACT, "32016R0679"])

    assert [(a.id, a.celex) for a in actors] == [("member_state", None)]


def test_a_cross_cutting_subject_keeps_no_single_definition():
    """Pointing at one act's definition would misrepresent the other's."""
    llm = _llm([DefinedSubject(label="Member State", defined_in=f"{AI_ACT}_003.003")])
    actors = generate(_graph(), llm, acts=[AI_ACT, "32016R0679"])

    assert actors[0].defined_in is None


def test_dump_is_byte_identical_across_runs():
    """A diff must mean a real change, so the serialisation cannot depend on ordering luck."""
    actors = [
        Actor(id="provider", label="Provider", celex=AI_ACT),
        Actor(id="deployer", label="Deployer", celex=AI_ACT),
    ]
    assert dumps(actors) == dumps(list(reversed(actors)))


def test_dump_and_load_round_trip():
    actors = [
        Actor(id="provider", label="Provider", celex=AI_ACT, defined_in=f"{AI_ACT}_003.003"),
        Actor(id="provider_of_a_high_risk_ai_system",
              label="Provider of a high-risk AI system", celex=AI_ACT, is_a=["provider"]),
    ]
    assert loads(dumps(actors)) == actors
