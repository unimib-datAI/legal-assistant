"""Source inventory for CJEU judgments — what the HTML contains, layer by layer.

A judgment reaches the graph through two transformations, and each is checked separately:

1. **HTML → tree**: ``parse_case_law`` turns the markup into sections and body items.
   The inventory here is :func:`html_fragments`, taken from the parser's own
   ``_linearize`` so the comparison is against the same reading of the document.
2. **Tree → graph**: ``split_paragraphs`` turns a section's body into ``CaseLawParagraph``
   nodes. The inventory is :func:`body_fragments`, and the exemptions are the items the
   splitter is *designed* to consume.

Both work on HTML already in memory. Nothing here fetches.
"""
from __future__ import annotations

import logging
from typing import Iterable, List

from bs4 import BeautifulSoup

from legal_assistant.case_law.html_parser import _find_topics, _linearize
from legal_assistant.case_law.kg_builder import (
    PREAMBLE_SECTIONS,
    _OPERATIVE_ANCHOR,
    _OPERATIVE_NUM,
    _PARAGRAPH_NUM,
    _SIGNATURES,
)
from legal_assistant.validation.plan import GraphPlan

logger = logging.getLogger(__name__)


# ── layer 1: HTML → tree ─────────────────────────────────────────────────────

def html_fragments(html: str) -> List[str]:
    """Every text fragment the parser reads out of the document, in order."""
    return [text for _, text in _linearize(BeautifulSoup(html, "html.parser"))]


def html_exemptions(html: str) -> List[str]:
    """Fragments deliberately not carried into the tree verbatim.

    Only the index block: ``_split_topics`` breaks it into individual CaseLawTopic labels,
    so the block as a whole never appears anywhere.
    """
    topics = _find_topics(BeautifulSoup(html, "html.parser"))
    return [topics] if topics else []


def tree_fragments(flat: Iterable[dict]) -> List[str]:
    """Every heading and body item present in the parsed tree."""
    fragments: List[str] = []
    for section in flat:
        fragments.append(section["heading"])
        fragments.extend(section.get("body", []))
    return fragments


# ── layer 2: tree → graph ────────────────────────────────────────────────────

def _strip_number(item: str) -> str:
    """Remove the leading paragraph number, as ``split_paragraphs`` does before storing."""
    for pattern in (_PARAGRAPH_NUM, _OPERATIVE_NUM):
        match = pattern.match(item)
        if match:
            return item[match.end():].strip()
    return item.strip()


def body_fragments(flat: Iterable[dict]) -> List[str]:
    """Body items that must end up inside a ``CaseLawParagraph``.

    Takes the flattened tree — the same ``flat`` the builder receives — so the inventory and
    the build are guaranteed to be looking at the same document.

    Preamble sections are excluded: ``_write_sections`` gives them a node but deliberately
    writes no paragraphs for them, because they carry EUR-Lex boilerplate rather than
    judicial reasoning.
    """
    fragments: List[str] = []
    for section in flat:
        if section["heading"] in PREAMBLE_SECTIONS:
            continue
        fragments.extend(_strip_number(item) for item in section.get("body", []))
    return fragments


def body_exemptions(flat: Iterable[dict]) -> List[str]:
    """Body items ``split_paragraphs`` is designed to consume rather than store.

    The "hereby rules" line is a marker that switches the splitter into operative mode, and
    a bare "[Signatures]" closes the judgment. Neither is content.

    An unnumbered item with no preceding numbered paragraph is also dropped by the splitter
    — a headnote fragment with nothing to attach to.
    """
    exempt: List[str] = []
    for section in flat:
        if section["heading"] in PREAMBLE_SECTIONS:
            continue
        pending = False
        for item in section.get("body", []):
            stripped = _strip_number(item)
            if _OPERATIVE_ANCHOR.search(item) or _SIGNATURES.match(item.strip()):
                exempt.append(stripped)
                pending = True
                continue
            if _PARAGRAPH_NUM.match(item) or _OPERATIVE_NUM.match(item):
                pending = True
            elif not pending:
                exempt.append(stripped)
    return exempt


# ── reconstruction from the plan ─────────────────────────────────────────────

def reconstructed_fragments(plan: GraphPlan) -> List[str]:
    """Headings and paragraph text the plan would write, walked in DFS order."""
    fragments: List[str] = []
    for celex in plan.roots("CaseLaw"):
        for _, node in plan.dfs(celex):
            if node.label == "CaseLawSection":
                fragments.append(node.properties.get("heading", ""))
            elif node.label == "CaseLawParagraph":
                fragments.append(node.properties.get("text", ""))
    # Topics hang off CaseLaw directly, not through the containment DFS.
    fragments += [n.properties.get("label", "")
                  for n in plan.node_ops if n.label == "CaseLawTopic"]
    return fragments


def paragraph_texts(plan: GraphPlan) -> List[str]:
    """Only the ``CaseLawParagraph`` text — the target of the layer-2 check."""
    return [n.properties.get("text", "") for n in plan.node_ops
            if n.label == "CaseLawParagraph"]
