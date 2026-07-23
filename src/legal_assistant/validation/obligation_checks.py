"""Pure checks over an obligations plan, run through the same gate the acts use.

A wrong obligation is worse than a missing one: it reads as a duty someone owes, and a
compliance checklist is trusted precisely because nobody re-reads the regulation behind it.
Each function takes what it needs and returns violations, so each is testable against a
hand-built plan.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Set

from legal_assistant.obligations.models import Actor, source_id_of
from legal_assistant.validation.checks import Violation
from legal_assistant.validation.plan import GraphPlan

# Edges from an obligation to the vocabulary. Both must land on a defined actor.
_ACTOR_RELATIONS = ("ADDRESSED_TO", "BENEFITS")


def obligations_anchored(plan: GraphPlan, passage_ids: Set[str]) -> List[Violation]:
    """Every obligation must come from a passage the graph actually holds.

    The source pipeline segments on its own paragraph ids, and six of those are duplicated in
    the AI Act markup by amending articles. Joining on them naively attaches Article 108's
    obligations to Article 17(3), so the anchor is checked rather than assumed.
    """
    violations: List[Violation] = []
    for node in plan.node_ops:
        if node.label != "Obligation":
            continue
        try:
            source = source_id_of(node.id)
        except ValueError as exc:
            violations.append(Violation("obligation_bad_id", node.id, str(exc)))
            continue
        if source not in passage_ids:
            violations.append(Violation(
                "obligation_unanchored", node.id,
                f"extracted from {source!r}, which is not a passage in the graph",
            ))
    return violations


def hierarchy_terminates(actors: Sequence[Actor]) -> List[Violation]:
    """``IS_A`` must be acyclic and closed.

    The role filter walks ``IS_A*0..``, so a cycle is an unbounded traversal at query time,
    and a parent nobody defined silently stops the walk short of duties it should have
    reached.
    """
    by_id = {actor.id: actor for actor in actors}
    violations: List[Violation] = []

    for actor in actors:
        for parent in actor.is_a:
            if parent not in by_id:
                violations.append(Violation(
                    "actor_unknown_parent", actor.id,
                    f"qualifies {parent!r}, which no actor defines",
                ))

    for actor in actors:
        seen = {actor.id}
        frontier = [parent for parent in actor.is_a if parent in by_id]
        while frontier:
            current = frontier.pop()
            if current in seen:
                violations.append(Violation(
                    "actor_cycle", actor.id, f"IS_A re-enters {current!r}"))
                break
            seen.add(current)
            frontier.extend(p for p in by_id[current].is_a if p in by_id)

    return violations


def actor_targets_exist(plan: GraphPlan, actors: Iterable[Actor]) -> List[Violation]:
    """Every ``ADDRESSED_TO`` and ``BENEFITS`` must land on an actor in the vocabulary.

    An edge to an actor that does not exist is an obligation nobody can be filtered onto: it
    is in the graph and unreachable, which is worse than absent because it inflates counts.
    """
    known = {actor.id for actor in actors}
    return [
        Violation(
            "actor_not_in_vocabulary", edge.left_id,
            f"-[:{edge.rel_type}]-> {edge.right_id!r}, which is not in the vocabulary",
        )
        for edge in plan.edge_ops
        if edge.rel_type in _ACTOR_RELATIONS and edge.right_id not in known
    ]


def unmapped_within_ceiling(unmapped: int, total: int, ceiling: float) -> List[Violation]:
    """Too many addressee strings anchoring to nothing means the vocabulary is incomplete.

    The fix is to regenerate, not to hand-edit: a high share says the generation missed a
    subject the acts define. Zero of zero is not a breach, it is an empty run, which is a
    different failure and reported elsewhere.
    """
    if total <= 0:
        return []
    share = unmapped / total
    if share <= ceiling:
        return []
    return [Violation(
        "addressees_unmapped", "",
        f"{unmapped}/{total} addressee strings ({share:.1%}) matched no actor, "
        f"ceiling is {ceiling:.1%}",
    )]
