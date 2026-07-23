"""The gate must block: a corrupted plan raises, and nothing reaches the real graph."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from legal_assistant.graph.recorder import RecordingGraph
from legal_assistant.validation.checks import (
    conflicting_upserts,
    conservation,
    containment_is_tree,
    dangling_edges,
    depth_and_labels,
)
from legal_assistant.validation.gate import (
    GraphValidationError,
    build_and_write,
    build_plan,
    build_validated,
)


def build_good(graph) -> None:
    """A minimal, well-formed act: Act -> Chapter -> Article -> Paragraph, plus a Recital."""
    graph.upsert_graph_node("Act", {"id": "ACT", "title": "An act"})
    graph.upsert_graph_node("Chapter", {"id": "ACTcpt_I", "number": "I", "title": "Ch"})
    graph.create_relationship("Act", "Chapter", "ACT", "ACTcpt_I", "CONTAINS")
    graph.upsert_graph_node("Article", {"id": "ACTart_1", "title": "Art", "text": "body"})
    graph.create_relationship("Chapter", "Article", "ACTcpt_I", "ACTart_1", "CONTAINS")
    graph.upsert_graph_node("Paragraph", {"id": "ACT_001.001", "text": "the paragraph text"})
    graph.create_relationship("Article", "Paragraph", "ACTart_1", "ACT_001.001", "CONTAINS")
    graph.upsert_graph_node("Recital", {"id": "ACTrct_1", "number": "1", "text": "a recital"})
    graph.create_relationship("Act", "Recital", "ACT", "ACTrct_1", "CONTAINS")


def test_good_plan_passes():
    plan = build_validated(build_good, "ACT", label="good")
    assert len(plan.nodes) == 5
    assert [n.id for _, n in plan.dfs("ACT")] == [
        "ACT", "ACTcpt_I", "ACTart_1", "ACT_001.001", "ACTrct_1",
    ]


def test_dangling_edge_is_caught():
    def build(graph):
        build_good(graph)
        # An article that cites a paragraph nobody created: today Neo4j swallows this.
        graph.create_relationship("Article", "Paragraph", "ACTart_1", "ACT_999.001", "CONTAINS")

    plan = build_plan(build)
    kinds = {v.kind for v in dangling_edges(plan)}
    assert kinds == {"dangling_edge"}
    with pytest.raises(GraphValidationError, match="dangling_edge"):
        build_validated(build, "ACT")


def test_duplicate_section_is_caught():
    def build(graph):
        build_good(graph)
        graph.upsert_graph_node("Chapter", {"id": "ACTcpt_II", "number": "II", "title": "Ch2"})
        graph.create_relationship("Act", "Chapter", "ACT", "ACTcpt_II", "CONTAINS")
        # The same article contained by two different chapters.
        graph.create_relationship("Chapter", "Article", "ACTcpt_II", "ACTart_1", "CONTAINS")

    plan = build_plan(build)
    assert {v.kind for v in containment_is_tree(plan, "ACT")} == {"multiple_parents"}


def test_removed_paragraph_is_caught_by_conservation():
    source = ["the paragraph text", "a paragraph the parser dropped"]
    plan = build_plan(build_good)
    reconstructed = [n.properties.get("text", "") for _, n in plan.dfs("ACT")]

    violations = conservation(source, reconstructed)
    assert [v.kind for v in violations] == ["text_lost"]
    assert "dropped" in violations[0].detail


def test_conservation_is_clean_when_nothing_is_lost():
    plan = build_plan(build_good)
    reconstructed = [n.properties.get("text", "") for _, n in plan.dfs("ACT")]
    assert conservation(["the paragraph text", "a recital"], reconstructed) == []


def test_conservation_flags_duplicates():
    """One source fragment stored as two nodes is as wrong as one lost."""
    assert [v.kind for v in conservation(["alpha"], ["alpha", "alpha"])] == ["text_duplicated"]
    # Two source fragments, two nodes: correct.
    assert conservation(["alpha", "alpha"], ["alpha", "alpha"]) == []


def test_conservation_ignores_substring_coincidence():
    """A short fragment inside longer node text is presence, not duplication.

    Topic labels like "EU law" legitimately occur inside dozens of paragraphs; counting
    substring hits would report them all as duplicated.
    """
    assert conservation(
        ["EU law"],
        ["EU law", "a paragraph about EU law", "another on EU law generally"],
    ) == []


def test_conflicting_upsert_is_caught():
    def build(graph):
        build_good(graph)
        graph.upsert_graph_node("Article", {"id": "ACTart_1", "title": "A different title"})

    violations = conflicting_upserts(build_plan(build))
    assert [v.kind for v in violations] == ["conflicting_upsert"]
    assert "title" in violations[0].detail


def test_bad_label_transition_is_caught():
    def build(graph):
        build_good(graph)
        graph.create_relationship("Paragraph", "Article", "ACT_001.001", "ACTart_1", "CONTAINS")

    assert [v.kind for v in depth_and_labels(build_plan(build))] == ["bad_transition"]


def test_orphan_is_caught():
    def build(graph):
        build_good(graph)
        graph.upsert_graph_node("Chapter", {"id": "ORPHAN", "number": "X", "title": "?"})
        graph.upsert_graph_node("Article", {"id": "ORPHANart_1", "title": "?", "text": "?"})
        graph.create_relationship("Chapter", "Article", "ORPHAN", "ORPHANart_1", "CONTAINS")

    kinds = [v.kind for v in containment_is_tree(build_plan(build), "ACT")]
    assert kinds.count("orphan") == 2


def test_nothing_is_written_when_validation_fails():
    """The point of the gate: a failed build must not reach the database."""
    def build(graph):
        build_good(graph)
        graph.create_relationship("Article", "Paragraph", "ACTart_1", "MISSING", "CONTAINS")

    neo4j = MagicMock()
    with pytest.raises(GraphValidationError):
        build_and_write(build, neo4j, "ACT")

    neo4j.upsert_graph_node.assert_not_called()
    neo4j.create_relationship.assert_not_called()


def test_valid_plan_is_written():
    neo4j = MagicMock()
    build_and_write(build_good, neo4j, "ACT")
    assert neo4j.upsert_graph_node.call_count == 5
    assert neo4j.create_relationship.call_count == 4


def test_strict_false_degrades_to_a_warning():
    def build(graph):
        build_good(graph)
        graph.create_relationship("Article", "Paragraph", "ACTart_1", "MISSING", "CONTAINS")

    plan = build_validated(build, "ACT", strict=False)
    assert plan is not None      # returned anyway


def test_recorder_returns_the_node_id():
    """Builders use the return value of upsert_graph_node as the id."""
    recorder = RecordingGraph()
    assert recorder.upsert_graph_node("Act", {"id": "ACT"}) == "ACT"


def test_non_determinism_is_caught():
    counter = {"n": 0}

    def build(graph):
        counter["n"] += 1
        build_good(graph)
        graph.upsert_graph_node("Paragraph", {"id": f"ACT_001.{counter['n']:03d}",
                                              "text": "unstable"})
        graph.create_relationship("Article", "Paragraph", "ACTart_1",
                                  f"ACT_001.{counter['n']:03d}", "CONTAINS")

    with pytest.raises(GraphValidationError, match="non_deterministic"):
        build_validated(build, "ACT")


def test_fingerprint_ignores_emission_order():
    def build_reordered(graph):
        graph.upsert_graph_node("Recital", {"id": "ACTrct_1", "number": "1", "text": "a recital"})
        graph.upsert_graph_node("Act", {"id": "ACT", "title": "An act"})
        graph.upsert_graph_node("Paragraph", {"id": "ACT_001.001", "text": "the paragraph text"})
        graph.upsert_graph_node("Article", {"id": "ACTart_1", "title": "Art", "text": "body"})
        graph.upsert_graph_node("Chapter", {"id": "ACTcpt_I", "number": "I", "title": "Ch"})
        graph.create_relationship("Act", "Recital", "ACT", "ACTrct_1", "CONTAINS")
        graph.create_relationship("Article", "Paragraph", "ACTart_1", "ACT_001.001", "CONTAINS")
        graph.create_relationship("Chapter", "Article", "ACTcpt_I", "ACTart_1", "CONTAINS")
        graph.create_relationship("Act", "Chapter", "ACT", "ACTcpt_I", "CONTAINS")

    assert build_plan(build_good).fingerprint() == build_plan(build_reordered).fingerprint()


def test_fingerprint_changes_with_content():
    def build_changed(graph):
        build_good(graph)
        graph.upsert_graph_node("Paragraph", {"id": "ACT_001.002", "text": "extra"})
        graph.create_relationship("Article", "Paragraph", "ACTart_1", "ACT_001.002", "CONTAINS")

    assert build_plan(build_good).fingerprint() != build_plan(build_changed).fingerprint()
