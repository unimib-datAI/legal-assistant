"""Batch-ingest CJEU case law into the knowledge graph.

The judgments to ingest are read straight out of the graph: the act loader already
creates a ``(:CaseLaw)`` stub per judgment listed as "Interpreted by" in the EUR-Lex
metadata, together with the ``(:CaseLaw)-[:INTERPRETS]->(:Article|:Paragraph|:Chapter)``
edge. Those stubs *are* the set of judgments interpreting our acts, so this module needs
no separate case list — but it does require :mod:`legal_assistant.pipelines.graph_build`
to have run first.

Summaries are optional: they cost one LLM call per section and paragraph-level retrieval
does not use them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from legal_assistant import config
from legal_assistant.case_law.html_parser import CaseLawHTMLError, parse_celex
from legal_assistant.case_law.kg_builder import build_from_tree, celex_to_case_number
from legal_assistant.case_law.tree import flatten
from legal_assistant.graph.client import Neo4jGraph
from legal_assistant.graph.queries import CaseLawQueries
from legal_assistant.resources import make_embeddings

logger = logging.getLogger(__name__)

CASE_LAW_NODE_LABEL = "CaseLawParagraph"

# GDPR: by far the most litigated of the four acts, so the useful default.
DEFAULT_ACTS: tuple[str, ...] = ("32016R0679",)


@dataclass
class IngestTotals:
    """Counts for one ingest run, plus the judgments that could not be fetched."""

    judgments: int = 0
    sections: int = 0
    paragraphs: int = 0
    operative: int = 0
    failed: List[Tuple[str, str]] = field(default_factory=list)


def resolve_celex_list(
    graph: Neo4jGraph,
    acts: Sequence[str],
    celex: Sequence[str] | None = None,
    limit: int | None = None,
) -> List[str]:
    """Explicit ``celex`` ids if given, otherwise every judgment interpreting ``acts``."""
    if celex:
        return [c.strip().upper() for c in celex]

    rows = graph.query(CaseLawQueries.GET_CASE_LAW_BY_ACTS, {"acts": list(acts)})
    celex_list = [row["celex"] for row in rows]
    return celex_list[:limit] if limit else celex_list


def _summarise(celex: str, flat: List[dict]) -> List[dict]:
    # Imported lazily: summarisation is off by default and pulls in the LLM stack.
    from legal_assistant.case_law.llm_orchestrator import summarize_section

    logger.info("Summarising %s (%d sections)…", celex, len(flat))
    return [s for s in (summarize_section(section) for section in flat) if s]


def ingest(graph: Neo4jGraph, celex_list: Sequence[str], with_summaries: bool = False) -> IngestTotals:
    """Parse and write each judgment. A judgment that cannot be fetched is skipped, not fatal."""
    totals = IngestTotals()

    for i, celex in enumerate(celex_list, start=1):
        label = f"{celex} ({celex_to_case_number(celex)})"
        try:
            roots = parse_celex(celex)
        except CaseLawHTMLError as exc:
            # Judgments before ~2012 have no XHTML manifestation in Cellar, only FMX.
            logger.warning("[%d/%d] SKIP %s — %s", i, len(celex_list), label, exc)
            totals.failed.append((celex, str(exc)))
            continue
        except OSError as exc:
            logger.warning("[%d/%d] SKIP %s — fetch failed: %s", i, len(celex_list), label, exc)
            totals.failed.append((celex, f"fetch failed: {exc}"))
            continue

        summaries = _summarise(celex, flatten(roots)) if with_summaries else None
        counts = build_from_tree(celex, roots, graph, summaries=summaries)

        totals.judgments += 1
        totals.sections += counts["sections"]
        totals.paragraphs += counts["paragraphs"]
        totals.operative += counts["operative"]
        logger.info(
            "[%d/%d] %s — %d sections, %d paragraphs (%d operative)",
            i, len(celex_list), label,
            counts["sections"], counts["paragraphs"], counts["operative"],
        )

    return totals


def embed_and_index(graph: Neo4jGraph) -> None:
    """Embed the ingested judgment paragraphs and (re)create their vector index."""
    embeddings = make_embeddings()
    graph.generate_text_embeddings(
        embed_fn=embeddings.embed_documents,
        embedding_dim=config.EMBEDDING_DIM,
        node_name=CASE_LAW_NODE_LABEL,
    )
    graph.create_vector_index(CASE_LAW_NODE_LABEL, CASE_LAW_NODE_LABEL, config.EMBEDDING_DIM)


def delete_existing_content(graph: Neo4jGraph) -> None:
    """Drop ingested sections/paragraphs; the ``(:CaseLaw)`` stubs and INTERPRETS edges stay."""
    logger.info("Deleting existing case law content (CaseLaw nodes and their INTERPRETS edges are kept)…")
    graph.query(CaseLawQueries.DELETE_CASE_LAW_CONTENT)
