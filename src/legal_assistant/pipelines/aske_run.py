"""Run the ASKE topic-extraction cycle over the paragraphs already in the graph.

This is phase 2 of the system (see .claude/CLAUDE.md): seed-based classification plus
iterative terminology enrichment, writing the resulting Concept nodes back onto the
paragraphs. Lifted out of the Streamlit page so it can also run from the CLI; the
function returns its results rather than writing a report file, leaving that to callers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from legal_assistant.graph.seed import SEEDS
from legal_assistant.resources import make_graph_client
from legal_assistant.text.preprocessor import TextPreprocessor
from legal_assistant.topic.aske import ASKETopicExtractor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AskeParams:
    """Tunables of the ASKE cycle. Defaults match the values the UI has always shown."""

    n_generations: int = 15
    alpha: float = 0.4  # classification threshold
    beta: float = 0.4  # enrichment threshold
    gamma: int = 7  # max new terms per concept


@dataclass
class AskeResult:
    """Concepts found and how many paragraphs were re-tagged."""

    active: List[Dict[str, Any]] = field(default_factory=list)
    inactive: List[Dict[str, Any]] = field(default_factory=list)
    updated_paragraphs: int = 0

    def as_report(self) -> List[Dict[str, Any]]:
        """The active concepts as a JSON-serialisable label + sorted-terms listing."""
        return [
            {"label": concept["label"], "terms": sorted(_term_labels(concept))}
            for concept in sorted(self.active, key=lambda c: c["label"])
        ]


def _term_labels(concept: Dict[str, Any]) -> set:
    """Terms are stored either as plain strings or as ``{"label": ...}`` dicts."""
    return {t["label"] if isinstance(t, dict) else t for t in concept.get("terms", [])}


def run_aske(params: AskeParams | None = None, top_n_topics: int = 3) -> AskeResult:
    """Run one full ASKE cycle and write the per-paragraph topics back to Neo4j."""
    params = params or AskeParams()

    graph = make_graph_client()
    aske = ASKETopicExtractor(graph)
    preprocessor = TextPreprocessor()

    paragraphs = graph.get_paragraphs_from_kg()
    chunks = preprocessor.to_chunks(paragraphs, skip_first=True)
    logger.info("[aske] %d paragraph chunk(s) to classify", len(chunks))

    concepts, final_classifications = aske.run_aske_cycle(
        chunks=chunks,
        seeds=SEEDS,
        n_generations=params.n_generations,
        alpha=params.alpha,
        beta=params.beta,
        gamma=params.gamma,
    )

    paragraph_topics = aske.aggregate_topics_by_paragraph(final_classifications, top_n=top_n_topics)
    updated = graph.update_paragraph_topics(paragraph_topics)

    result = AskeResult(
        active=[c for c in concepts if c.get("active", True)],
        inactive=[c for c in concepts if not c.get("active", True)],
        updated_paragraphs=updated,
    )
    logger.info(
        "[aske] %d active concept(s), %d inactive, %d paragraph(s) updated",
        len(result.active), len(result.inactive), result.updated_paragraphs,
    )
    return result
