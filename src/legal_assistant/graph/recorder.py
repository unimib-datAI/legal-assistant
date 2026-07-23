"""An in-memory stand-in for :class:`~legal_assistant.graph.client.Neo4jGraph`.

The graph builders (``GraphLoader``, ``kg_builder.create_case_law_kg``) talk to Neo4j
through exactly two methods: ``upsert_graph_node`` and ``create_relationship``. Handing
them a :class:`RecordingGraph` instead of the real client makes them build the whole graph
in memory, where it can be inspected as a whole before a single write reaches the database.

Nothing here touches the network. The builders are not modified: this is duck typing on
the only part of the interface they actually use.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeOp:
    """One recorded ``upsert_graph_node`` call."""

    label: str
    properties: Dict[str, Any]

    @property
    def id(self) -> str:
        return self.properties["id"]


@dataclass(frozen=True)
class EdgeOp:
    """One recorded ``create_relationship`` call."""

    left_label: str
    right_label: str
    left_id: str
    right_id: str
    rel_type: str


class RecordingGraph:
    """Records graph operations instead of writing them.

    Mirrors the signatures of ``Neo4jGraph.upsert_graph_node`` and
    ``Neo4jGraph.create_relationship``, including the return value of the former, which the
    builders use as the node id.
    """

    def __init__(self) -> None:
        self.node_ops: List[NodeOp] = []
        self.edge_ops: List[EdgeOp] = []

    def upsert_graph_node(self, node_name: str, node_properties: Dict[str, Any]) -> str:
        """Record a node upsert and return its id, as the real client does."""
        self.node_ops.append(NodeOp(label=node_name, properties=dict(node_properties)))
        return node_properties["id"]

    def create_relationship(
        self,
        left_node_name: str,
        right_node_name: str,
        left_id: str,
        right_id: str,
        relationship: str,
    ) -> None:
        """Record a relationship. Unlike the real client, nothing is silently dropped."""
        self.edge_ops.append(EdgeOp(
            left_label=left_node_name,
            right_label=right_node_name,
            left_id=left_id,
            right_id=right_id,
            rel_type=relationship,
        ))

    def __repr__(self) -> str:
        return f"RecordingGraph(nodes={len(self.node_ops)}, edges={len(self.edge_ops)})"
