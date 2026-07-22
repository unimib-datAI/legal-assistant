"""The blocking gate: build into a recorder, validate, and only then replay.

This is the entry point the builders' callers use. Nothing reaches Neo4j unless every check
passes, so a parser regression fails loudly at build time instead of quietly corrupting the
graph.
"""
from __future__ import annotations

import logging
from typing import Callable, Iterable, List, Optional, Sequence

from legal_assistant.graph.recorder import RecordingGraph
from legal_assistant.validation.checks import Violation, conservation, structural
from legal_assistant.validation.plan import GraphPlan

logger = logging.getLogger(__name__)

# A builder: given something that quacks like Neo4jGraph, it writes a graph into it.
BuildFn = Callable[[RecordingGraph], None]


class GraphValidationError(RuntimeError):
    """Raised when a plan fails validation. Carries the full report."""

    def __init__(self, label: str, violations: Sequence[Violation]) -> None:
        self.label = label
        self.violations: List[Violation] = list(violations)
        super().__init__(self.report())

    def report(self) -> str:
        lines = [f"{self.label}: {len(self.violations)} validation violation(s)"]
        lines += [f"  {v}" for v in self.violations]
        return "\n".join(lines)


def build_plan(build_fn: BuildFn) -> GraphPlan:
    """Run ``build_fn`` against a recorder and return what it would have written."""
    recorder = RecordingGraph()
    build_fn(recorder)
    return GraphPlan.from_recorder(recorder)


def build_validated(
    build_fn: BuildFn,
    root_id: str,
    *,
    label: str = "graph",
    source_inventory: Optional[Iterable[str]] = None,
    reconstructed: Optional[Callable[[GraphPlan], Iterable[str]]] = None,
    exempt: Iterable[str] = (),
    conservation_kind: str = "text",
    check_determinism: bool = True,
    strict: bool = True,
) -> GraphPlan:
    """Build, validate, and return the plan. Nothing is written to any real graph here.

    ``source_inventory`` + ``reconstructed`` enable the conservation check: the first is the
    list of fragments the source document contains, the second extracts the corresponding
    fragments from the built plan. Omitting them runs the structural checks only.

    ``check_determinism`` re-runs ``build_fn`` on a second recorder and compares
    fingerprints. It is free — the same bytes go in, and nothing touches the network.

    With ``strict=False`` violations are logged as warnings and the plan is returned anyway.
    """
    plan = build_plan(build_fn)
    violations = structural(plan, root_id)

    if source_inventory is not None and reconstructed is not None:
        violations += conservation(
            source_inventory, reconstructed(plan), exempt, kind=conservation_kind
        )

    if check_determinism:
        second = build_plan(build_fn)
        if second.fingerprint() != plan.fingerprint():
            violations.append(Violation(
                "non_deterministic", root_id,
                "two builds over the same input produced different fingerprints "
                f"({plan.fingerprint()[:12]}… vs {second.fingerprint()[:12]}…)",
            ))

    if not violations:
        logger.info("[gate] %s: valid (%s)", label, plan)
        return plan

    if strict:
        raise GraphValidationError(label, violations)

    error = GraphValidationError(label, violations)
    logger.warning("[gate] %s", error.report())
    return plan


def build_and_write(
    build_fn: BuildFn,
    graph,
    root_id: str,
    **kwargs,
) -> GraphPlan:
    """:func:`build_validated`, then replay onto ``graph``.

    The only function in the project that turns a validated plan into database writes.
    """
    plan = build_validated(build_fn, root_id, **kwargs)
    plan.replay(graph)
    return plan
