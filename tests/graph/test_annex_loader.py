"""Annexes must reach the graph as nodes the rest of the system can address.

Ids are positional and always valid; the citation a lawyer writes, "Annex III, point 1(a)",
is composed for display and never used for lookup. That split is the whole reason annex ids
are not parsed out of the prose numbering, which is the failure the removed ``CITES`` edges
demonstrated.
"""
from __future__ import annotations

import pathlib
from unittest.mock import patch

import pytest

from legal_assistant import config
from legal_assistant.graph.loader import GraphLoader
from legal_assistant.graph.recorder import RecordingGraph
from legal_assistant.scraper.eurlex_exporter import EURLexHTMLParser
from legal_assistant.validation.checks import containment_is_tree, depth_and_labels
from legal_assistant.validation.plan import GraphPlan

AI_ACT = "32024R1689"


def _plan(celex: str) -> GraphPlan:
    path = pathlib.Path(config.CORPUS_DIR) / f"{celex}.html"
    if not path.is_file():
        pytest.skip(f"{celex}.html not in corpus/")

    with patch.object(EURLexHTMLParser, "_get_case_law", return_value=[]):
        data = EURLexHTMLParser(str(path), celex, "", "").extract_data()

    recorder = RecordingGraph()
    GraphLoader(recorder)._emit(data)
    return GraphPlan.from_recorder(recorder)


@pytest.fixture(scope="module")
def plan() -> GraphPlan:
    return _plan(AI_ACT)


def _nodes(plan: GraphPlan, label: str):
    return [node for node in plan.node_ops if node.label == label]


def test_every_annex_becomes_a_node(plan):
    ids = [node.id for node in _nodes(plan, "Annex")]
    assert ids[:3] == ["32024R1689anx_I", "32024R1689anx_II", "32024R1689anx_III"]
    assert len(ids) == 13


def test_annex_point_ids_are_positional(plan):
    """Position, not prose numbering: an id must be valid whatever the markup says."""
    ids = [node.id for node in _nodes(plan, "AnnexPoint")
           if node.id.startswith("32024R1689anx_VI.")]
    assert ids == [f"32024R1689anx_VI.p_{n:03d}" for n in range(1, 5)]


def test_point_label_composes_the_citation(plan):
    """Article 6(2) cites "Annex III, point 1(a)", so that is what the label must read."""
    labels = [node.properties["point_label"] for node in _nodes(plan, "AnnexPoint")
              if node.id.startswith("32024R1689anx_III.")]
    assert "III, point 1(a)" in labels


def test_unlabelled_points_have_no_citation_label(plan):
    """Lead-in prose carries no marker, so it is cited through its annex, not by point."""
    labels = [node.properties["point_label"] for node in _nodes(plan, "AnnexPoint")
              if node.id.startswith("32024R1689anx_III.")]
    assert None in labels


def test_annexes_hang_off_the_act(plan):
    """Same structural position as recitals: the act contains them directly."""
    edges = {(e.left_label, e.rel_type, e.right_label) for e in plan.edge_ops}
    assert ("Act", "CONTAINS", "Annex") in edges
    assert ("Annex", "CONTAINS", "AnnexPoint") in edges


def test_annex_transitions_are_allowed(plan):
    """A label transition the schema does not declare means a builder changed shape."""
    assert depth_and_labels(plan) == []


def test_containment_stays_a_tree(plan):
    assert containment_is_tree(plan, AI_ACT) == []


def test_loading_is_deterministic(plan):
    assert _plan(AI_ACT).fingerprint() == plan.fingerprint()
