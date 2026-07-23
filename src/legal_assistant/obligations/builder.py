"""Turn analysed obligations into graph nodes and edges.

Each :class:`AnalysedObligation` becomes one ``Obligation`` node hanging off the passage it
came from, its addressees and beneficiaries resolved to ``Actor`` edges through anchoring.
The source output's element lists are reduced to the node's single fields, and each field
keeps the weakest method among its values so the node's trust is honest.

Pure over the graph client: it writes through ``upsert_graph_node`` / ``create_relationship``,
so a ``RecordingGraph`` captures exactly what would reach Neo4j and the obligation checks run
on it before anything is written for real.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from legal_assistant.obligations.anchoring import AnchorResult
from legal_assistant.obligations.models import (
    Actor,
    AnalysedObligation,
    ExtractedElement,
    ExtractionMethod,
    Modality,
    source_id_of,
    weakest_of,
)

logger = logging.getLogger(__name__)

# An annex point's id carries this marker; a paragraph's does not. Used to label the STATES
# edge with the right source node type.
_ANNEX_MARKER = "anx_"


def _source_label(source_id: str) -> str:
    return "AnnexPoint" if _ANNEX_MARKER in source_id else "Paragraph"


def _join(elements: Sequence[ExtractedElement]) -> Optional[str]:
    values = [e.value.strip() for e in elements if e.value and e.value.strip()]
    return "; ".join(values) if values else None


def _method(elements: Sequence[ExtractedElement]) -> ExtractionMethod:
    return weakest_of(e.method for e in elements)


def _modality(obligation: AnalysedObligation) -> Modality:
    """A prohibition reads as the negated predicate; everything else is an obligation."""
    predicate = (obligation.predicate.value or "").lower()
    negated = any(neg in predicate for neg in (" not ", "not ", "neither", "shall not", "must not"))
    return Modality.PROHIBITION if negated else Modality.OBLIGATION


def _properties(obligation: AnalysedObligation) -> dict:
    methods = {
        "addressee_method": _method(obligation.addressees),
        "predicate_method": obligation.predicate.method,
        "target_method": _method(obligation.targets),
        "specification_method": _method(obligation.specifications),
        "precondition_method": _method(obligation.preconditions),
        "beneficiary_method": _method(obligation.beneficiaries),
    }
    populated = [m for m in methods.values() if m is not ExtractionMethod.NONE]
    weakest = weakest_of(populated) if populated else ExtractionMethod.NONE

    return {
        "id": obligation.id,
        "celex": obligation.celex,
        "modality": _modality(obligation).value,
        "obligation_type": obligation.obligation_type.value,
        "predicate_text": obligation.predicate.value,
        "predicate_voice": obligation.predicate.verb,
        "target": _join(obligation.targets),
        "specification": _join(obligation.specifications),
        "precondition": _join(obligation.preconditions),
        "beneficiary_text": _join(obligation.beneficiaries),
        "weakest_method": weakest.value,
        **{name: method.value for name, method in methods.items()},
    }


def _resolved_ids(elements: Sequence[ExtractedElement], anchor: AnchorResult) -> List[str]:
    """Actor ids the element values resolved to, in order, deduplicated."""
    ids: List[str] = []
    for element in elements:
        for actor_id in anchor.resolved.get(element.value or "", []):
            if actor_id not in ids:
                ids.append(actor_id)
    return ids


def _write_actor(graph, actor: Actor) -> None:
    graph.upsert_graph_node("Actor", {
        "id": actor.id,
        "label": actor.label,
        "celex": actor.celex,
        "defined_in": actor.defined_in,
    })
    for parent in actor.is_a:
        graph.create_relationship("Actor", "Actor", actor.id, parent, "IS_A")


def build_obligations(
    graph,
    obligations: List[AnalysedObligation],
    anchor: AnchorResult,
    existing_actors: Sequence[Actor],
) -> None:
    """Write the obligation subgraph: actors, obligation nodes, and their edges.

    The whole vocabulary is written, not only the promoted actors, because the graph never
    held the vocabulary as nodes, only the ``actors.yaml`` file. Without this an ADDRESSED_TO
    edge points at an actor that does not exist and Neo4j's ``MERGE`` drops it in silence.
    Upserts are idempotent, so re-running an ingest is safe.
    """
    for actor in list(existing_actors) + list(anchor.promoted):
        _write_actor(graph, actor)

    for obligation in obligations:
        graph.upsert_graph_node("Obligation", _properties(obligation))

        source = source_id_of(obligation.id)
        graph.create_relationship(
            _source_label(source), "Obligation", source, obligation.id, "STATES")

        for actor_id in _resolved_ids(obligation.addressees, anchor):
            graph.create_relationship(
                "Obligation", "Actor", obligation.id, actor_id, "ADDRESSED_TO")

        for actor_id in _resolved_ids(obligation.beneficiaries, anchor):
            graph.create_relationship(
                "Obligation", "Actor", obligation.id, actor_id, "BENEFITS")

    logger.info(
        "[builder] wrote %d obligation(s) and %d promoted actor(s)",
        len(obligations), len(anchor.promoted),
    )
