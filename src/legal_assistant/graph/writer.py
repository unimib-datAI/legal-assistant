"""The write interface the graph builders talk to, and its transactional implementation.

``GraphLoader`` and ``kg_builder`` reach Neo4j through exactly two methods,
``upsert_graph_node`` and ``create_relationship``. That narrow surface is what lets the same
builder run against the real client, against :class:`~legal_assistant.graph.recorder.RecordingGraph`
for validation, and against an open transaction here. :class:`GraphWriter` gives the
interface a name so the three stay in step.

:class:`TransactionalGraph` wraps a *runner*, meaning anything exposing ``run``. A neo4j
``Session`` and a neo4j ``Transaction`` both qualify, so ``Neo4jGraph`` delegates its own two
methods here rather than keeping a second copy of the statements: one home per statement.

Writing a whole act or judgment through a single transaction is what makes an interrupted
run safe to resume. Without it a crash leaves a half-written unit that still looks present,
and the resume check has no way to tell it from a complete one.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Protocol, runtime_checkable

from legal_assistant.graph.queries import NodeQueries, RelationQueries

logger = logging.getLogger(__name__)


@runtime_checkable
class GraphWriter(Protocol):
    """The two calls every graph builder makes."""

    def upsert_graph_node(self, node_name: str, node_properties: Dict[str, Any]) -> str: ...

    def create_relationship(
        self,
        left_node_name: str,
        right_node_name: str,
        left_id: str,
        right_id: str,
        relationship: str,
    ) -> None: ...


class TransactionalGraph:
    """A :class:`GraphWriter` that runs its statements on one session or transaction."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    def upsert_graph_node(self, node_name: str, node_properties: Dict[str, Any]) -> str:
        """Create or update a node, returning its id as the builders expect."""
        query = NodeQueries.CREATE_NODE.format(node_name=node_name)
        result = self._runner.run(query, node_properties=node_properties)
        return result.single()["node_id"]

    def create_relationship(
        self,
        left_node_name: str,
        right_node_name: str,
        left_id: str,
        right_id: str,
        relationship: str,
    ) -> None:
        """Relate two existing nodes. Silently writes nothing if either is missing."""
        query = RelationQueries.CREATE_RELATIONSHIP.format(
            left_node_name=left_node_name,
            right_node_name=right_node_name,
            relationship=relationship,
        )
        self._runner.run(query, left_id=left_id, right_id=right_id)
