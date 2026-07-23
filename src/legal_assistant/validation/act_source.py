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

# Text nodes that carry the substance of an act. Headings and numbering use other classes.
_TEXT_CLASS = "oj-normal"


def _fragments(soup: BeautifulSoup, div_id: re.Pattern) -> List[str]:
    return [
        text
        for div in soup.find_all("div", id=div_id)
        for p in div.find_all("p", class_=_TEXT_CLASS)
        if (text := p.get_text(separator=" ", strip=True))
    ]


def html_fragments(html_path: str | Path) -> List[str]:
    """Every substantive text fragment in the act's articles and recitals."""
    soup = BeautifulSoup(Path(html_path).read_text(encoding="utf-8"), "html.parser")
    return _fragments(soup, _ARTICLE_DIV) + _fragments(soup, _RECITAL_DIV)


def article_fragments(html_path: str | Path) -> List[str]:
    """Only the article side, the half the paragraph splitter is responsible for."""
    soup = BeautifulSoup(Path(html_path).read_text(encoding="utf-8"), "html.parser")
    return _fragments(soup, _ARTICLE_DIV)


def reconstructed_fragments(plan: GraphPlan) -> List[str]:
    """Paragraph and recital text from the plan, never ``Article.text``.

    Walked off the recorded nodes rather than the DFS so that a paragraph orphaned by a
    broken containment edge still counts as *written*: losing it is a structural failure,
    reported by ``containment_is_tree``, not a conservation one.
    """
    return [
        node.properties.get("text", "")
        for node in plan.node_ops
        if node.label in ("Paragraph", "Recital")
    ]
