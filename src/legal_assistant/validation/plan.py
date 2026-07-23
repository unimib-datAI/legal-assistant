"""The recorded graph, as a structure that can be walked, fingerprinted, and replayed.

A :class:`GraphPlan` is everything a builder *would* have written, held in memory. It is
the object every check runs against, and the only thing that ever gets replayed onto a real
Neo4j connection.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Iterator, List, Sequence, Set, Tuple

from legal_assistant.graph.recorder import EdgeOp, NodeOp, RecordingGraph

logger = logging.getLogger(__name__)

# Properties excluded from the fingerprint: large, derived, and not part of the structure
# the fingerprint is meant to pin.
_VOLATILE_PROPERTIES = frozenset({"textEmbedding"})

# The edges that form the containment hierarchies. Everything else (INTERPRETS, HAS_TOPIC)
# is a cross-reference, not structure.
CONTAINMENT_RELATIONS = ("CONTAINS", "HAS_SECTION", "HAS_PARAGRAPH")


class GraphPlan:
    """A recorded graph: nodes keyed by id, edges in emission order."""

    def __init__(self, node_ops: Sequence[NodeOp], edge_ops: Sequence[EdgeOp]) -> None:
        self.node_ops: List[NodeOp] = list(node_ops)
        self.edge_ops: List[EdgeOp] = list(edge_ops)

        # Last write wins, mirroring Neo4j's `SET n +=`. Conflicting writes are reported by
        # `checks.conflicting_upserts`, not silently resolved here.
        self.nodes: Dict[str, NodeOp] = {op.id: op for op in self.node_ops}

    @classmethod
    def from_recorder(cls, recorder: RecordingGraph) -> "GraphPlan":
        return cls(recorder.node_ops, recorder.edge_ops)

    # ── traversal ────────────────────────────────────────────────────────────

    def children(self, node_id: str, rel_types: Sequence[str] = CONTAINMENT_RELATIONS) -> List[str]:
        """Ids reachable from ``node_id`` along ``rel_types``, in emission order."""
        wanted = set(rel_types)
        return [e.right_id for e in self.edge_ops
                if e.left_id == node_id and e.rel_type in wanted]

    def dfs(
        self,
        root_id: str,
        rel_types: Sequence[str] = CONTAINMENT_RELATIONS,
    ) -> Iterator[Tuple[int, NodeOp]]:
        """Pre-order DFS from ``root_id``, yielding ``(depth, node)``.

        A node already visited is not descended into again: a cycle would otherwise not
        terminate. Detecting that revisit is `checks.containment_is_tree`'s job, not this
        one's; here it only guarantees termination.
        """
        seen: Set[str] = set()
        stack: List[Tuple[int, str]] = [(0, root_id)]
        while stack:
            depth, node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            node = self.nodes.get(node_id)
            if node is None:
                continue  # dangling: reported by checks.dangling_edges
            yield depth, node
            # Reversed so that emission order is preserved on the way down.
            for child_id in reversed(self.children(node_id, rel_types)):
                stack.append((depth + 1, child_id))

    def roots(self, label: str) -> List[str]:
        """Ids of every recorded node carrying ``label``."""
        return [op.id for op in self.node_ops if op.label == label]

    # ── identity ─────────────────────────────────────────────────────────────

    def fingerprint(self) -> str:
        """sha256 of a canonical serialisation: order-insensitive, content-sensitive.

        Two runs over the same bytes must produce the same fingerprint even if the builder
        emits in a different order; any change to a node property or an edge must change it.
        """
        nodes = [
            {
                "label": op.label,
                "id": op.id,
                "properties": {k: v for k, v in sorted(op.properties.items())
                               if k not in _VOLATILE_PROPERTIES},
            }
            for op in sorted(self.nodes.values(), key=lambda o: (o.label, o.id))
        ]
        edges = sorted(
            [e.left_label, e.rel_type, e.right_label, e.left_id, e.right_id]
            for e in set(self.edge_ops)
        )
        canonical = json.dumps({"nodes": nodes, "edges": edges},
                               sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ── output ───────────────────────────────────────────────────────────────

    def replay(self, graph: Any) -> None:
        """Write the plan to a real graph client, in the order it was recorded.

        Nodes first, then edges: the real ``CREATE_RELATIONSHIP`` matches both endpoints, so
        an edge emitted before its node would silently write nothing.
        """
        for op in self.node_ops:
            graph.upsert_graph_node(node_name=op.label, node_properties=op.properties)
        for e in self.edge_ops:
            graph.create_relationship(
                left_node_name=e.left_label,
                right_node_name=e.right_label,
                left_id=e.left_id,
                right_id=e.right_id,
                relationship=e.rel_type,
            )
        logger.info("[plan] replayed %d node(s), %d edge(s)",
                    len(self.node_ops), len(self.edge_ops))

    def __repr__(self) -> str:
        return f"GraphPlan(nodes={len(self.nodes)}, edges={len(self.edge_ops)})"
