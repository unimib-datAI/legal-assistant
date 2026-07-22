"""Pure checks over a :class:`~legal_assistant.validation.plan.GraphPlan`.

Every function takes a plan (plus whatever it needs) and returns a list of
:class:`Violation`. No I/O, no side effects — so each is testable against a hand-built plan.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from legal_assistant.validation.plan import CONTAINMENT_RELATIONS, GraphPlan

# Label transitions the loaders are allowed to emit, read off the builders themselves
# (graph/loader.py, case_law/kg_builder.py) and cross-checked against the live schema in
# docs/knowledge-graph.md. An edge outside this set means a builder changed shape.
ALLOWED_TRANSITIONS: Set[Tuple[str, str, str]] = {
    ("Act", "CONTAINS", "Chapter"),
    ("Act", "CONTAINS", "Recital"),
    ("Chapter", "CONTAINS", "Section"),
    ("Chapter", "CONTAINS", "Article"),
    ("Section", "CONTAINS", "Article"),
    ("Article", "CONTAINS", "Paragraph"),
    ("CaseLaw", "HAS_SECTION", "CaseLawSection"),
    ("CaseLaw", "HAS_TOPIC", "CaseLawTopic"),
    ("CaseLawSection", "CONTAINS", "CaseLawSection"),
    ("CaseLawSection", "HAS_PARAGRAPH", "CaseLawParagraph"),
    # Cross-references, not structure: a judgment interprets a provision.
    ("CaseLaw", "INTERPRETS", "Article"),
    ("CaseLaw", "INTERPRETS", "Paragraph"),
    ("CaseLaw", "INTERPRETS", "Chapter"),
}

# Edges whose right-hand endpoint is expected to live in another document's plan, so a
# missing endpoint is not this plan's fault. INTERPRETS is emitted while loading an act and
# points at provisions of that same act, but the CaseLaw stub itself is created here.
_CROSS_DOCUMENT_RELATIONS = frozenset({"INTERPRETS"})

_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


@dataclass(frozen=True)
class Violation:
    """One problem found in a plan."""

    kind: str
    node_id: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.node_id}: {self.detail}"


def normalise(text: str) -> str:
    """Canonical form for text comparison: NFKC, lowercase, alphanumerics only.

    Aggressive on purpose — the question a conservation check answers is "did this content
    survive", not "was the whitespace preserved". Punctuation and spacing differ freely
    between the source markup and the concatenated node text.
    """
    text = unicodedata.normalize("NFKC", text)
    return _NON_ALNUM.sub("", _WHITESPACE.sub(" ", text).lower())


# ── structural checks ────────────────────────────────────────────────────────

def dangling_edges(plan: GraphPlan) -> List[Violation]:
    """Edges whose endpoint was never recorded as a node.

    The real `CREATE_RELATIONSHIP` does `MATCH … MATCH … MERGE`, so a missing endpoint
    writes nothing and raises nothing. Here it is an error.
    """
    violations = []
    for e in plan.edge_ops:
        if e.rel_type in _CROSS_DOCUMENT_RELATIONS:
            continue
        for side, node_id, label in (("left", e.left_id, e.left_label),
                                     ("right", e.right_id, e.right_label)):
            if node_id not in plan.nodes:
                violations.append(Violation(
                    "dangling_edge", node_id,
                    f"{e.left_id} -[:{e.rel_type}]-> {e.right_id}: "
                    f"{side} endpoint ({label}) was never created",
                ))
    return violations


def conflicting_upserts(plan: GraphPlan) -> List[Violation]:
    """The same id written twice with different values for a shared key.

    Neo4j's `SET n +=` overwrites silently, so today the second write wins and nobody knows.
    """
    violations = []
    seen: Dict[str, Dict[str, object]] = {}
    for op in plan.node_ops:
        previous = seen.get(op.id)
        if previous is None:
            seen[op.id] = dict(op.properties)
            continue
        for key, value in op.properties.items():
            if key in previous and previous[key] != value:
                violations.append(Violation(
                    "conflicting_upsert", op.id,
                    f"property {key!r} rewritten: {previous[key]!r} -> {value!r}",
                ))
        previous.update(op.properties)
    return violations


def containment_is_tree(
    plan: GraphPlan,
    root_id: str,
    rel_types: Sequence[str] = CONTAINMENT_RELATIONS,
) -> List[Violation]:
    """The containment edges from ``root_id`` must form a tree.

    Every structural node reachable exactly once: no cycles, no double parents. Nodes that
    carry containment edges but are not reachable from the root are orphans.
    """
    violations: List[Violation] = []

    parents: Dict[str, List[str]] = {}
    wanted = set(rel_types)
    for e in plan.edge_ops:
        if e.rel_type in wanted:
            parents.setdefault(e.right_id, []).append(e.left_id)

    for node_id, ps in parents.items():
        distinct = sorted(set(ps))
        if len(distinct) > 1:
            violations.append(Violation(
                "multiple_parents", node_id, f"reached from {distinct}"))
        elif len(ps) > 1:
            violations.append(Violation(
                "duplicate_edge", node_id, f"contained by {distinct[0]} {len(ps)} times"))

    reached = {node.id for _, node in plan.dfs(root_id, rel_types)}

    # A node involved in containment but never reached from the root is detached.
    involved = set(parents) | {e.left_id for e in plan.edge_ops if e.rel_type in wanted}
    for node_id in sorted(involved - reached):
        if node_id in plan.nodes:
            violations.append(Violation(
                "orphan", node_id,
                f"not reachable from {root_id} along {sorted(wanted)}"))

    # A cycle shows up as a node whose parent chain re-enters itself; DFS stops there, so
    # the tell is an edge into an already-reached node from outside the tree.
    for e in plan.edge_ops:
        if e.rel_type in wanted and e.right_id == root_id:
            violations.append(Violation(
                "cycle", root_id, f"root is contained by {e.left_id}"))

    return violations


def depth_and_labels(
    plan: GraphPlan,
    allowed: Set[Tuple[str, str, str]] = ALLOWED_TRANSITIONS,
) -> List[Violation]:
    """Every edge must be a label transition the schema allows."""
    violations = []
    for e in plan.edge_ops:
        transition = (e.left_label, e.rel_type, e.right_label)
        if transition not in allowed:
            violations.append(Violation(
                "bad_transition", e.left_id,
                f"({e.left_label})-[:{e.rel_type}]->({e.right_label}) is not an allowed "
                f"transition (target {e.right_id})",
            ))
    return violations


# ── conservation ─────────────────────────────────────────────────────────────

def conservation(
    source: Iterable[str],
    reconstructed: Iterable[str],
    exempt: Iterable[str] = (),
    *,
    kind: str = "text",
    sample: int = 8,
) -> List[Violation]:
    """Every source fragment must appear in the reconstructed graph exactly once.

    Both directions matter: a fragment consumed twice is as wrong as one lost. ``exempt``
    holds fragments the builder deliberately drops or synthesises, which must be declared
    rather than tolerated silently.

    Presence is matched by containment, because a source fragment is one ``<p>`` while a
    node's text may concatenate several of them. Duplication is matched by *exact* node
    text instead: counting substring occurrences across the concatenated corpus is
    meaningless for short fragments — a topic label like "EU law" legitimately occurs inside
    dozens of paragraphs without having been stored twice.
    """
    exempt_norm = {normalise(e) for e in exempt if normalise(e)}
    source_counts = Counter(n for s in source if (n := normalise(s)) and n not in exempt_norm)
    reconstructed_norm = [n for r in reconstructed if (n := normalise(r))]
    blob = "".join(reconstructed_norm)
    exact_counts = Counter(reconstructed_norm)

    violations = []
    missing = [frag for frag in source_counts if frag not in blob]
    for frag in missing[:sample]:
        violations.append(Violation(
            f"{kind}_lost", "", f"source fragment not found in the graph: {frag[:90]}…"))
    if len(missing) > sample:
        violations.append(Violation(
            f"{kind}_lost", "", f"…and {len(missing) - sample} more lost fragment(s)"))

    duplicated = [(frag, exact_counts[frag]) for frag, n in source_counts.items()
                  if exact_counts[frag] > n]
    for frag, found in duplicated[:sample]:
        violations.append(Violation(
            f"{kind}_duplicated", "",
            f"stored as {found} separate nodes, {source_counts[frag]}x in the source: "
            f"{frag[:90]}…"))

    return violations


def structural(plan: GraphPlan, root_id: str) -> List[Violation]:
    """Every structural check, in one call."""
    return (
        dangling_edges(plan)
        + conflicting_upserts(plan)
        + depth_and_labels(plan)
        + containment_is_tree(plan, root_id)
    )
