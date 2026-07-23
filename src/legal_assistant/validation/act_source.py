"""Source inventory for EUR-Lex acts: what the HTML contains vs. what reaches the graph.

The source of truth is the published markup: every ``<p class="oj-normal">`` inside an
article or recital div is text a reader sees, so every one of them must survive into a
``Paragraph`` or ``Recital`` node.

**``Article.text`` is deliberately excluded from the reconstruction.** The exporter stores
the article's whole ``full_text``, which already contains all of its paragraphs, counting
it would make a dropped paragraph look present and defeat the check entirely.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup

from legal_assistant.validation.plan import GraphPlan

logger = logging.getLogger(__name__)

_ARTICLE_DIV = re.compile(r"^art_")
_RECITAL_DIV = re.compile(r"^rct_")
_ANNEX_DIV = re.compile(r"^anx_")

# Text nodes that carry the substance of an act. Headings and numbering use other classes.
_TEXT_CLASS = "oj-normal"

# Headings inside an annex: the annex's own two-line title, and its internal group headings.
# Excluded for the same reason ``oj-ti-art`` never enters the article inventory: they are
# navigation, and the loader carries them as ``Annex.title`` and ``AnnexPoint.section_heading``
# rather than as point text.
_ANNEX_HEADING_CLASSES = frozenset({"oj-doc-ti", "oj-ti-grseq-1"})


def _fragments(soup: BeautifulSoup, div_id: re.Pattern) -> List[str]:
    return [
        text
        for div in soup.find_all("div", id=div_id)
        for p in div.find_all("p", class_=_TEXT_CLASS)
        if (text := p.get_text(separator=" ", strip=True))
    ]


def _cell_own_text(cell) -> str:
    """A table cell's own text, excluding anything belonging to a table nested inside it.

    Walks strings rather than elements: annexes wrap cell prose in ``<p>`` in some places and
    in ``<span>`` in others, and a cell holding sub-points contains whole nested tables whose
    text belongs to those sub-points, not to this one.
    """
    owner = cell.find_parent("table")
    parts = [
        text
        for node in cell.find_all(string=True)
        if node.find_parent("table") is owner and (text := node.strip())
    ]
    return " ".join(parts)


def _annex_fragments(soup: BeautifulSoup) -> List[str]:
    """Every text fragment inside the act's annexes.

    Annexes cannot be inventoried the way articles are. Their text lives in table cells, and
    Annex VI of the AI Act carries no ``oj-normal`` class anywhere, so keying on that class
    would count zero fragments there, reconstruct zero, and let the gate pass on an annex that
    never reached the graph.

    Cells and loose paragraphs are collected separately and never overlap: a paragraph inside
    a cell is already covered by that cell's own text. Numbering fragments such as "1." or
    "(a)" are left in; they are matched by containment and so pass trivially, and excluding
    them would need a classifier that cannot tell a marker from a genuinely short provision,
    of which Annex II's list of offences has several.
    """
    fragments: List[str] = []
    for div in soup.find_all("div", id=_ANNEX_DIV):
        for cell in div.find_all("td"):
            if text := _cell_own_text(cell):
                fragments.append(text)

        for p in div.find_all("p"):
            if p.find_parent("td") is not None:
                continue
            if _ANNEX_HEADING_CLASSES & set(p.get("class") or []):
                continue
            if text := p.get_text(separator=" ", strip=True):
                fragments.append(text)

    return fragments


def annex_fragments(html_path: str | Path) -> List[str]:
    """Only the annex side. Empty for the three acts in the corpus that have no annexes."""
    soup = BeautifulSoup(Path(html_path).read_text(encoding="utf-8"), "html.parser")
    return _annex_fragments(soup)


def html_fragments(html_path: str | Path) -> List[str]:
    """Every substantive text fragment in the act's articles, recitals and annexes."""
    soup = BeautifulSoup(Path(html_path).read_text(encoding="utf-8"), "html.parser")
    return (
        _fragments(soup, _ARTICLE_DIV)
        + _fragments(soup, _RECITAL_DIV)
        + _annex_fragments(soup)
    )


def article_fragments(html_path: str | Path) -> List[str]:
    """Only the article side, the half the paragraph splitter is responsible for."""
    soup = BeautifulSoup(Path(html_path).read_text(encoding="utf-8"), "html.parser")
    return _fragments(soup, _ARTICLE_DIV)


def reconstructed_fragments(plan: GraphPlan) -> List[str]:
    """Paragraph, recital and annex-point text from the plan, never ``Article.text``.

    Walked off the recorded nodes rather than the DFS so that a paragraph orphaned by a
    broken containment edge still counts as *written*: losing it is a structural failure,
    reported by ``containment_is_tree``, not a conservation one.
    """
    return [
        node.properties.get("text", "")
        for node in plan.node_ops
        if node.label in ("Paragraph", "Recital", "AnnexPoint")
    ]
