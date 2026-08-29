"""Shared pytest fixtures.

Unit tests must not touch Neo4j or the OpenAI API: mock those boundaries with
``unittest.mock`` (see .claude/CLAUDE.md). Test modules mirror the package layout:
``src/legal_assistant/rag/acts.py`` -> ``tests/rag/test_acts.py``.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def graph_mock() -> MagicMock:
    """A stand-in ``Neo4jGraph`` whose ``transaction()`` yields the mock itself.

    Builders write a whole unit through ``with graph.transaction() as tx``, so on a bare
    ``MagicMock`` the writes would land on an anonymous child mock and every assertion about
    ``upsert_graph_node`` would silently read zero. Folding the transaction back onto the
    mock keeps those assertions both true and readable.
    """
    graph = MagicMock()
    graph.transaction.return_value.__enter__.return_value = graph
    return graph
