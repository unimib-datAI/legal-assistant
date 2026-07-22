"""Build the act knowledge graph: scrape EUR-Lex, load Neo4j, embed, index.

This is phase 1 of the system (see .claude/CLAUDE.md). It used to live inline in the
Streamlit page, which made it unreachable from a script; the page now calls
:func:`build_graph` and only renders progress.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

from legal_assistant import config
from legal_assistant.graph.loader import GraphLoader
from legal_assistant.resources import make_embeddings, make_graph_client
from legal_assistant.scraper.eurlex_document_utils import EurlexDocumentUtils

logger = logging.getLogger(__name__)

# Node labels that carry an embedding + vector index. Ordered as they are built.
EMBEDDED_LABELS: tuple[str, ...] = ("Paragraph", "Recital", "Article")

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


def build_graph(
    celex_ids: Sequence[str], clear_db: bool = True, *, strict: bool = True
) -> GraphBuildResult:
    """Load ``celex_ids`` from EUR-Lex into Neo4j, then embed and index their text nodes.

    Every act is fetched, built in memory and **validated before anything is written** — and
    before the database is cleared. A parser regression therefore leaves the existing graph
    untouched rather than wiping it and failing halfway through the reload.

    ``clear_db`` wipes the database once all acts have validated. ``strict=False`` downgrades
    validation failures to warnings. The connection is always closed, even on error.
    """
    ids = [c.strip() for c in celex_ids if c.strip()]
    if not ids:
        raise ValueError("No CELEX ids given.")

    graph = make_graph_client()
    try:
        graph.verify_connection()

        eurlex_utils = EurlexDocumentUtils()
        loader = GraphLoader(graph)
        configs = [eurlex_utils.build_document_config(c) for c in ids]

        # Validate everything first: nothing below this line runs if an act is malformed.
        plans = loader.plan_all_documents(configs, strict=strict)
        logger.info("[graph_build] %d/%d act(s) validated", len(plans), len(ids))

        if clear_db:
            graph.clear_database()

        for celex, plan in plans:
            loader.write(plan)
            logger.info("[graph_build] wrote %s", celex)

        embeddings = make_embeddings()
        for label in EMBEDDED_LABELS:
            dimension = graph.generate_text_embeddings(
                embed_fn=embeddings.embed_documents,
                embedding_dim=config.EMBEDDING_DIM,
                node_name=label,
            )
            graph.create_vector_index(label, label, dimension)
            logger.info("[graph_build] %s embedded and indexed (dim=%d)", label, dimension)

        return GraphBuildResult(celex_ids=ids, indexed_labels=list(EMBEDDED_LABELS))
    finally:
        graph.close()
