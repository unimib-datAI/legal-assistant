"""Build the act knowledge graph: scrape EUR-Lex, load Neo4j, embed, index.

This is phase 1 of the system (see .claude/CLAUDE.md). It used to live inline in the
Streamlit page, which made it unreachable from a script; the page now calls
:func:`build_graph` and only renders progress.

A run is resumable. Acts already in the graph are skipped before anything is downloaded, so
relaunching an interrupted build costs nothing for the work already done. That rests on
:meth:`legal_assistant.graph.client.Neo4jGraph.transaction`: an act is written inside one
transaction, so it is either wholly present or wholly absent, never half loaded and
mistaken for finished.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from legal_assistant import config
from legal_assistant.graph.client import Neo4jGraph
from legal_assistant.graph.completion import act_is_loaded
from legal_assistant.graph.loader import GraphLoader
from legal_assistant.resources import make_embeddings, make_graph_client
from legal_assistant.scraper.eurlex_document_utils import EurlexDocumentUtils
from legal_assistant.validation.gate import GraphValidationError

logger = logging.getLogger(__name__)

# Node labels that carry an embedding + vector index. Ordered as they are built.
# Only the AI Act has annex points, so the AnnexPoint index stays empty for the other three.
EMBEDDED_LABELS: tuple[str, ...] = ("Paragraph", "Recital", "Article", "AnnexPoint")

# The four acts the project models, in the order the graph is normally built.
DEFAULT_CELEX_IDS: tuple[str, ...] = (
    "32016R0679",  # GDPR
    "32024R1689",  # AI Act
    "32023R2854",  # Data Act
    "32022R0868",  # Data Governance Act
)


@dataclass(frozen=True)
class GraphBuildResult:
    """What a build produced, for the caller to report."""

    celex_ids: List[str]
    indexed_labels: List[str]
    skipped: List[str] = field(default_factory=list)
    failed: List[Tuple[str, str]] = field(default_factory=list)


def build_graph(
    celex_ids: Sequence[str],
    clear_db: bool = False,
    *,
    strict: bool = True,
    force: bool = False,
) -> GraphBuildResult:
    """Load ``celex_ids`` from EUR-Lex into Neo4j, then embed and index their text nodes.

    Two paths, and they differ on purpose.

    **Resumable (the default).** Each act is checked against the graph, and one already
    present is skipped before it is downloaded. The rest are planned, validated and written
    one at a time, each inside its own transaction. An act that fails to parse is recorded
    and the others still land.

    **Clean rebuild (``clear_db=True``).** Every act is fetched, built in memory and
    validated *before anything is written*, and before the database is cleared. A parser
    regression therefore leaves the existing graph untouched rather than wiping it and
    failing halfway through the reload. Presence is not consulted, since the wipe makes
    every act absent, and ``force`` is accepted without effect.

    ``strict=False`` downgrades validation failures to warnings. The connection is always
    closed, even on error.
    """
    ids = [c.strip() for c in celex_ids if c.strip()]
    if not ids:
        raise ValueError("No CELEX ids given.")

    graph = make_graph_client()
    try:
        graph.verify_connection()
        loader = GraphLoader(graph)

        if clear_db:
            written, skipped, failed = _rebuild(graph, loader, ids, strict=strict)
        else:
            written, skipped, failed = _resume(graph, loader, ids, strict=strict, force=force)

        _embed_and_index(graph)
        return GraphBuildResult(
            celex_ids=written,
            indexed_labels=list(EMBEDDED_LABELS),
            skipped=skipped,
            failed=failed,
        )
    finally:
        graph.close()


def _rebuild(
    graph: Neo4jGraph, loader: GraphLoader, ids: List[str], *, strict: bool
) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """Validate every act, then clear, then write. Nothing below the plan runs on failure."""
    eurlex_utils = EurlexDocumentUtils()
    configs = [eurlex_utils.build_document_config(c) for c in ids]

    plans = loader.plan_all_documents(configs, strict=strict)
    logger.info("[graph_build] %d/%d act(s) validated", len(plans), len(ids))

    graph.clear_database()

    for celex, plan in plans:
        loader.write(plan)
        logger.info("[graph_build] wrote %s", celex)

    return [celex for celex, _ in plans], [], []


def _resume(
    graph: Neo4jGraph, loader: GraphLoader, ids: List[str], *, strict: bool, force: bool
) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """Skip what is already loaded, process the rest one act at a time."""
    eurlex_utils = EurlexDocumentUtils()
    written: List[str] = []
    skipped: List[str] = []
    failed: List[Tuple[str, str]] = []

    for celex in ids:
        # Before the download, not after: the fetch is the expensive half of an act.
        if not force and act_is_loaded(graph, celex):
            logger.info("[graph_build] %s already loaded, skipping", celex)
            skipped.append(celex)
            continue

        try:
            document_config = eurlex_utils.build_document_config(celex)
            plan = loader.plan_document(document_config, strict=strict)
        except (GraphValidationError, OSError, RuntimeError, ValueError) as exc:
            logger.error("[graph_build] SKIP %s: %s", celex, exc)
            failed.append((celex, str(exc)))
            continue

        loader.write(plan)
        written.append(celex)
        logger.info("[graph_build] wrote %s", celex)

    return written, skipped, failed


def _embed_and_index(graph: Neo4jGraph) -> None:
    """Embed the act text nodes and (re)create their vector indexes.

    Naturally resumable: the underlying query only returns nodes without an embedding, so a
    run where nothing new was written costs one query per label.
    """
    embeddings = make_embeddings()
    for label in EMBEDDED_LABELS:
        dimension = graph.generate_text_embeddings(
            embed_fn=embeddings.embed_documents,
            embedding_dim=config.EMBEDDING_DIM,
            node_name=label,
        )
        graph.create_vector_index(label, label, dimension)
        logger.info("[graph_build] %s embedded and indexed (dim=%d)", label, dimension)
