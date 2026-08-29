"""The write seam: one implementation of the two write statements, three users of it.

``RecordingGraph``, the real client and ``TransactionalGraph`` are interchangeable because
the builders only ever call ``upsert_graph_node`` and ``create_relationship``. These tests
pin that interchangeability, and pin the transaction semantics the resume logic depends on:
a unit that was interrupted must leave nothing behind, so that "the node is present" can be
read as "the unit completed".
"""
from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from legal_assistant.graph.client import Neo4jGraph
from legal_assistant.graph.recorder import RecordingGraph
from legal_assistant.graph.writer import GraphWriter, TransactionalGraph


def _runner(node_id: str = "n1") -> MagicMock:
    """A stand-in for a neo4j Session or Transaction, both of which expose ``run``."""
    runner = MagicMock()
    runner.run.return_value.single.return_value = {"node_id": node_id}
    return runner


# ── the protocol ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("implementation", [RecordingGraph, Neo4jGraph, TransactionalGraph])
def test_every_writer_satisfies_the_protocol(implementation):
    """A fourth implementor added without both methods must fail here, not in production."""
    assert issubclass(implementation, GraphWriter)


# ── statements ───────────────────────────────────────────────────────────────

def test_upsert_runs_create_node_for_the_label():
    runner = _runner()
    TransactionalGraph(runner).upsert_graph_node("Paragraph", {"id": "p1", "text": "x"})

    query, kwargs = runner.run.call_args[0][0], runner.run.call_args[1]
    assert "MERGE (n:Paragraph {id: $node_properties.id})" in query
    assert kwargs["node_properties"] == {"id": "p1", "text": "x"}


def test_upsert_returns_the_node_id():
    """The builders use the return value as the node id, so it is part of the contract."""
    writer = TransactionalGraph(_runner(node_id="p1"))
    assert writer.upsert_graph_node("Paragraph", {"id": "p1"}) == "p1"


def test_create_relationship_runs_the_statement_for_both_labels():
    runner = _runner()
    TransactionalGraph(runner).create_relationship(
        left_node_name="Article",
        right_node_name="Paragraph",
        left_id="a1",
        right_id="p1",
        relationship="CONTAINS",
    )

    query, kwargs = runner.run.call_args[0][0], runner.run.call_args[1]
    assert "MATCH (ln:Article {id: $left_id})" in query
    assert "MATCH (rn:Paragraph {id: $right_id})" in query
    assert "MERGE (ln)-[:CONTAINS]->(rn)" in query
    assert kwargs == {"left_id": "a1", "right_id": "p1"}


# ── transaction semantics ────────────────────────────────────────────────────

def _graph_with_mock_driver() -> tuple[Neo4jGraph, MagicMock, MagicMock]:
    graph = Neo4jGraph.__new__(Neo4jGraph)  # no driver construction, no connection
    graph.driver = MagicMock()

    session = graph.driver.session.return_value.__enter__.return_value
    tx = session.begin_transaction.return_value
    return graph, session, tx


def test_transaction_commits_on_clean_exit():
    graph, _, tx = _graph_with_mock_driver()

    with graph.transaction() as writer:
        assert isinstance(writer, TransactionalGraph)

    tx.commit.assert_called_once()
    tx.rollback.assert_not_called()


def test_transaction_rolls_back_and_reraises_on_error():
    """The property the whole resume design rests on: an interrupted unit leaves nothing."""
    graph, _, tx = _graph_with_mock_driver()

    with pytest.raises(RuntimeError, match="boom"):
        with graph.transaction():
            raise RuntimeError("boom")

    tx.rollback.assert_called_once()
    tx.commit.assert_not_called()


def test_transaction_closes_the_session_on_both_paths():
    graph, _, _ = _graph_with_mock_driver()

    with graph.transaction():
        pass
    with pytest.raises(RuntimeError):
        with graph.transaction():
            raise RuntimeError("boom")

    assert graph.driver.session.return_value.__exit__.call_count == 2


def test_writes_inside_a_transaction_go_to_the_transaction():
    """Not to a fresh session, which is what made the old path non-atomic."""
    graph, session, tx = _graph_with_mock_driver()
    tx.run.return_value.single.return_value = {"node_id": "p1"}

    with graph.transaction() as writer:
        writer.upsert_graph_node("Paragraph", {"id": "p1"})

    tx.run.assert_called_once()
    session.run.assert_not_called()


# ── the real client delegates rather than duplicating the statements ─────────

def test_client_upsert_uses_one_session_and_the_shared_statement():
    graph, session, _ = _graph_with_mock_driver()
    session.run.return_value.single.return_value = {"node_id": "p1"}

    assert graph.upsert_graph_node("Paragraph", {"id": "p1"}) == "p1"

    assert "MERGE (n:Paragraph {id: $node_properties.id})" in session.run.call_args[0][0]


def test_client_create_relationship_uses_one_session_and_the_shared_statement():
    graph, session, _ = _graph_with_mock_driver()

    graph.create_relationship("Article", "Paragraph", "a1", "p1", "CONTAINS")

    assert "MERGE (ln)-[:CONTAINS]->(rn)" in session.run.call_args[0][0]
    assert session.run.call_args[1] == {"left_id": "a1", "right_id": "p1"}


def test_recorder_and_transactional_graph_agree_on_the_upsert_return_value():
    """Same contract, or a builder behaves differently depending on where it writes."""
    recorded = RecordingGraph().upsert_graph_node("Paragraph", {"id": "p1"})
    written = TransactionalGraph(_runner(node_id="p1")).upsert_graph_node("Paragraph", {"id": "p1"})
    assert recorded == written == "p1"


def test_recording_graph_offers_a_transaction_that_yields_itself():
    """``GraphLoader.write`` opens a transaction, and it is handed a recorder in validation.

    Recording is already atomic, so this is a shim rather than a mechanism, but without it
    every builder would need to know which kind of graph it is writing to.
    """
    recorder = RecordingGraph()
    with recorder.transaction() as writer:
        assert writer is recorder


def test_replay_accepts_any_writer():
    """``GraphPlan.replay`` is typed against the protocol, not against the real client."""
    from legal_assistant.validation.plan import GraphPlan

    source = RecordingGraph()
    source.upsert_graph_node("Article", {"id": "a1"})
    source.upsert_graph_node("Paragraph", {"id": "p1"})
    source.create_relationship("Article", "Paragraph", "a1", "p1", "CONTAINS")

    target = RecordingGraph()
    GraphPlan.from_recorder(source).replay(target)

    assert len(target.node_ops) == 2
    assert len(target.edge_ops) == 1
