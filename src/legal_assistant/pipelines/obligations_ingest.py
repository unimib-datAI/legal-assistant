"""Ingest deontic obligations into the graph.

The flow: detect candidate passages, filter each candidate sentence, analyse the ones that
carry a duty, anchor every addressee and beneficiary string onto the actor vocabulary, build
the obligation subgraph, validate it, and write it. The LLM stages are the expensive part;
the rest is deterministic.

`build_plan_from_obligations` is the deterministic core, split out so it can be validated
without a model: given analysed obligations it anchors, builds and validates, returning the
plan to write and the anchoring result to fold back into the vocabulary.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Callable, List, Sequence, Set, Tuple

from legal_assistant import config
from legal_assistant.obligations.analysis import analyse_sentence
from legal_assistant.obligations.anchoring import AnchorResult, anchor_strings
from legal_assistant.obligations.builder import build_obligations
from legal_assistant.obligations.concurrency import map_concurrent
from legal_assistant.obligations.detection import detect
from legal_assistant.obligations.filtering import filter_sentence, is_deontic
from legal_assistant.obligations.models import Actor, AnalysedObligation, ExtractedElement
from legal_assistant.obligations.references import citation_contents
from legal_assistant.obligations.vocabulary import load_vocabulary
from legal_assistant.validation.gate import build_plan
from legal_assistant.validation.checks import conflicting_upserts, dangling_edges
from legal_assistant.validation.obligation_checks import (
    actor_targets_exist,
    hierarchy_terminates,
    obligations_anchored,
    unmapped_within_ceiling,
)
from legal_assistant.validation.plan import GraphPlan

logger = logging.getLogger(__name__)

# Share of addressee strings allowed to match no actor before the load is refused.
UNMAPPED_CEILING = 0.25
# How often an unknown addressee must recur to be promoted to an actor of its own.
FREQUENCY_FLOOR = 3
# Minimum embedding similarity for a string to anchor to an existing actor.
MATCH_THRESHOLD = 0.6


def _party_strings(obligations: Sequence[AnalysedObligation]) -> Counter:
    """Every addressee and beneficiary value across the obligations, with its frequency."""
    counter: Counter = Counter()
    for obligation in obligations:
        for element in list(obligation.addressees) + list(obligation.beneficiaries):
            if element.value and element.value.strip():
                counter[element.value.strip()] += 1
    return counter


def build_plan_from_obligations(
    obligations: List[AnalysedObligation],
    actors: Sequence[Actor],
    passage_ids: Set[str],
    embed: Callable[[List[str]], List[Sequence[float]]],
    *,
    threshold: float = MATCH_THRESHOLD,
    frequency_floor: int = FREQUENCY_FLOOR,
    unmapped_ceiling: float = UNMAPPED_CEILING,
) -> Tuple[GraphPlan, AnchorResult]:
    """Anchor, build and validate. Raises if any obligation check fails.

    Returns the validated plan and the anchoring result, the latter so its promoted actors can
    be written back into the committed vocabulary by the caller.
    """
    strings = _party_strings(obligations)
    anchor = anchor_strings(strings, actors, embed,
                            threshold=threshold, frequency_floor=frequency_floor)

    def build(graph):
        build_obligations(graph, obligations, anchor, actors)

    plan = build_plan(build)

    all_actors = list(actors) + anchor.promoted
    total = sum(strings.values())
    unmapped = sum(anchor.unmapped.values())
    violations = (
        conflicting_upserts(plan)
        + dangling_edges(plan)
        + obligations_anchored(plan, passage_ids)
        + hierarchy_terminates(all_actors)
        + actor_targets_exist(plan, all_actors)
        + unmapped_within_ceiling(unmapped, total, unmapped_ceiling)
    )
    if violations:
        from legal_assistant.validation.gate import GraphValidationError

        raise GraphValidationError("obligations", violations)

    logger.info(
        "[obligations_ingest] built %d obligation(s), %d promoted actor(s), %d unmapped string(s)",
        len(obligations), len(anchor.promoted), unmapped,
    )
    return plan, anchor


def extract_obligations(
    graph, llm, acts: Sequence[str], limit: int | None = None,
    max_workers: int = config.EXTRACTION_MAX_WORKERS,
) -> List[AnalysedObligation]:
    """Run detection, filtering and analysis over the acts, returning analysed obligations."""
    detected = detect(graph, list(acts))
    if limit is not None:
        detected = detected[:limit]

    # The passage is the unit of parallelism: each is self-contained, so its per-passage
    # ordinal numbering stays correct however tasks interleave, and running passages
    # concurrently is what turns hours of sequential calls into minutes.
    per_passage = map_concurrent(
        lambda passage: _process_passage(llm, graph, passage),
        detected,
        max_workers=max_workers,
    )
    obligations = [ob for passage_obligations in per_passage for ob in passage_obligations]
    logger.info("[obligations_ingest] extracted %d obligation(s) from %d passage(s)",
                len(obligations), len(detected))
    return obligations


def _process_passage(llm, graph, passage) -> List[AnalysedObligation]:
    """Filter and analyse one passage's candidate sentences into obligations.

    Sequential within the passage: the ordinal runs across its sentences so two deontic
    sentences never number an obligation alike and overwrite each other in the graph.
    """
    obligations: List[AnalysedObligation] = []
    next_ordinal = 1
    for candidate in passage.candidates:
        result = filter_sentence(llm, sentence=candidate.sentence, context=passage.text)
        if not is_deontic(result.classification):
            continue
        citations = citation_contents(graph, candidate.references) or None
        extracted = analyse_sentence(
            llm, source_id=passage.par_id, celex=passage.celex,
            sentence=candidate.sentence, context=passage.text,
            citations=citations,
            start_ordinal=next_ordinal,
        )
        obligations.extend(extracted)
        next_ordinal += len(extracted)
    return obligations


def passage_ids_of(graph, acts: Sequence[str]) -> Set[str]:
    """Every Paragraph and AnnexPoint id of the acts, for the anchoring check."""
    from legal_assistant.graph.queries import NodeQueries

    rows = (
        graph.query(NodeQueries.GET_PARAGRAPHS_BY_ACTS, params={"acts": list(acts)})
        + graph.query(NodeQueries.GET_ANNEX_POINTS_BY_ACTS, params={"acts": list(acts)})
    )
    return {row["id"] for row in rows}


def delete_obligations(graph, acts: Sequence[str]) -> None:
    """Remove the obligations of the given acts, so an ingest can be re-run cleanly.

    ``DETACH DELETE`` drops each obligation with its STATES, ADDRESSED_TO and BENEFITS edges.
    Actor nodes are left in place; the ingest re-upserts them idempotently.
    """
    from legal_assistant.graph.queries import NodeQueries

    graph.query(NodeQueries.DELETE_OBLIGATIONS_BY_ACTS, params={"acts": list(acts)})
    logger.info("[obligations_ingest] deleted existing obligations for acts %s", list(acts))


def ingest(
    graph, llm, embed, acts: Sequence[str], limit: int | None = None,
    max_workers: int = config.EXTRACTION_MAX_WORKERS,
) -> AnchorResult:
    """Full ingestion against a live graph and model. Returns the anchoring result."""
    actors = load_vocabulary()
    obligations = extract_obligations(graph, llm, acts, limit=limit, max_workers=max_workers)
    passage_ids = passage_ids_of(graph, acts)

    plan, anchor = build_plan_from_obligations(obligations, actors, passage_ids, embed)
    plan.replay(graph)
    logger.info("[obligations_ingest] wrote obligation subgraph for acts %s", list(acts))
    return anchor
