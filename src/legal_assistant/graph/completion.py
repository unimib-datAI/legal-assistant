"""Has this unit already been ingested?

Both ingest pipelines ask before fetching, so that a run interrupted halfway resumes instead
of starting over. The answer comes from the graph itself rather than from a progress file:
each act and each judgment is written inside one transaction, so a unit is either wholly
present or wholly absent, and presence therefore means "complete and validated".
"""
from __future__ import annotations

import logging

from legal_assistant.graph.client import Neo4jGraph
from legal_assistant.graph.queries import CaseLawQueries, NodeQueries

logger = logging.getLogger(__name__)


def _first_done(rows: list[dict]) -> bool:
    """``done`` off the first row. No rows means nothing matched, so: not done."""
    return bool(rows[0]["done"]) if rows else False


def act_is_loaded(graph: Neo4jGraph, celex: str) -> bool:
    """True if ``celex`` is already in the graph as a complete act."""
    return _first_done(graph.query(NodeQueries.ACT_IS_LOADED, {"celex": celex}))


def case_law_is_ingested(graph: Neo4jGraph, celex: str) -> bool:
    """True if judgment ``celex`` already has its sections.

    Deliberately not a check on the ``(:CaseLaw)`` node: the act loader creates that stub for
    every "Interpreted by" reference it parses, well before the judgment is fetched, so the
    node is present for every judgment the corpus knows about and would answer yes to all of
    them.
    """
    return _first_done(graph.query(CaseLawQueries.CASE_LAW_IS_INGESTED, {"celex": celex}))
