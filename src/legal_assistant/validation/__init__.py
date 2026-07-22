"""Blocking, deterministic validation of the graph before it is written.

Nothing reaches Neo4j until the graph that is about to be written has been rebuilt with a
DFS and checked against the source document. See ``.claude/graph_validation.md``.
"""
from legal_assistant.validation.gate import GraphValidationError, build_validated
from legal_assistant.validation.plan import GraphPlan
from legal_assistant.validation.checks import Violation

__all__ = ["GraphPlan", "GraphValidationError", "Violation", "build_validated"]
