"""Structural parser for CJEU judgments, driven by the EUR-Lex XHTML markup.

EUR-Lex publishes every judgment with its hierarchy already encoded in CSS classes,
so the structure is *read*, not inferred. This replaces the docling + LLM + regex
pipeline in ``doc_parser.py``, which had to reconstruct that hierarchy from a
two-column PDF and got the depths wrong.

Two markup schemes exist; they share the same suffixes and differ only by a
``coj-`` prefix (modern, ~2020+) or its absence (legacy, pre-2016).

Depth is *relative to the document*: the same heading is ``sum-title-1`` in one
judgment and ``title-grseq-2`` in another. So the classes actually present after
the ``Judgment`` anchor are dense-ranked per document. The anchor itself is
excluded from that ranking; see ``_section_depths``.
"""
import logging
import re

import requests
from bs4 import BeautifulSoup, Tag

from legal_assistant.case_law.tree import Node, TreeBuilder

logger = logging.getLogger(__name__)

CELLAR_URL = "http://publications.europa.eu/resource/celex/{celex}"
_CELLAR_HEADERS = {
    "Accept": "application/xhtml+xml",
    "Accept-Language": "eng",
}

# Heading classes, with or without the modern "coj-" prefix.
_HEADING_CLASS = re.compile(r"^(?:coj-)?(?:sum-title-1|title-grseq-\d)$")
# The parenthetical keyword block ("Topics") that precedes the judgment body.
_INDEX_CLASS = re.compile(r"^(?:coj-)?index$")
# Footnotes ("* Language of the case: Dutch.") carry no content for the graph.
_NOISE_CLASS = re.compile(r"^(?:coj-)?note$")

_ANCHOR = "judgment"
_DASH = r"[-–—]"
_TOPIC_SPLIT = re.compile(r"\s+" + _DASH + r"\s*(?=[A-Z‘’'\"])")

# The index block always sits in parentheses; older judgments additionally wrap it in typographic
# quotes, newer ones do not:
#   legacy:  (‛Electronic communications — … — Articles 7 and 8’)
#   modern:  (Reference for a preliminary ruling – … – Concept of ‘consent’)
# Left in, the wrapping quotes fork the graph: ‛Personal data and Personal data become two distinct
# CaseLawTopic nodes, so the judgments sharing the corpus's most common topics stop linking.
#
# The trailing quote may only be dropped when the block *opens* with one. In the modern form the
# final ’ belongs to the last topic itself, and stripping it would corrupt "Concept of ‘consent’".
_OPEN_QUOTES = "‛‘“„«\"'"
_CLOSE_QUOTES = "’‘”»\"'"

PREAMBLE_SECTIONS = ("Reports of Cases", "Topics", "General Information")


class CaseLawHTMLError(RuntimeError):
    """The document could not be resolved or has no usable structure."""


def fetch_html(celex: str, timeout: int = 60) -> str:
    """Download the English XHTML manifestation of *celex* from Cellar.

    The eur-lex.europa.eu portal answers non-browser clients with an empty 202,
    so Cellar's content-negotiation endpoint is the only scriptable source.
    """
    url = CELLAR_URL.format(celex=celex.strip().upper())
    logger.info("Fetching case law XHTML: %s", url)
    response = requests.get(url, headers=_CELLAR_HEADERS, timeout=timeout)
    if response.status_code == 404:
        raise CaseLawHTMLError(
            f"No XHTML manifestation in Cellar for {celex}. "
            "Judgments before ~2012 are only published as FMX."
        )
    response.raise_for_status()
    response.encoding = "utf-8"
    return response.text


def _heading_class(tag: Tag) -> str | None:
    return next((c for c in (tag.get("class") or []) if _HEADING_CLASS.match(c)), None)


def _has_class(tag: Tag, pattern: re.Pattern[str]) -> bool:
    return any(pattern.match(c) for c in (tag.get("class") or []))


def _heading_rank(css_class: str) -> int:
    """Intrinsic ordering of a heading class, before per-document dense-ranking."""
    return 0 if css_class.endswith("sum-title-1") else int(css_class[-1])


def _text(tag: Tag) -> str:
    return " ".join(tag.get_text().split())


def _text_outside_tables(tag: Tag) -> str:
    """Text of *tag*, pruning any nested table.

    A numbered paragraph that introduces a quotation holds the quotation in a table nested
    inside its own prose cell. That inner table's rows are emitted separately, in document
    order, right after this one, so taking the cell's full ``get_text()`` here would repeat
    every quoted line inside the introducing paragraph.
    """
    parts: list[str] = []

    def walk(node: Tag) -> None:
        for child in node.children:
            name = getattr(child, "name", None)
            if name == "table":
                continue
            if name is None:
                parts.append(str(child))
            else:
                walk(child)

    walk(tag)
    return " ".join(" ".join(parts).split())


def _linearize(soup: BeautifulSoup) -> list[tuple[str | None, str]]:
    """Flatten the document into ordered (heading_class | None, text) pairs.

    Numbered paragraphs are split across two table cells, the number and the
    prose, and must be re-joined here. Without this merge every body paragraph
    would arrive as two fragments.

    The cells must be counted non-recursively. A paragraph that introduces a quotation
    ("3 Recitals 1, 4, 10 … state:") nests the quotation's own table inside its prose cell,
    so a recursive ``find_all("td")`` sees 20 cells rather than 2, the row fails the pair
    check, and the introducing paragraph is dropped from the document, while the quoted
    lines it introduces survive, orphaned.
    """
    items: list[tuple[str | None, str]] = []

    for element in soup.find_all(["p", "tr"]):
        if element.name == "tr":
            cells = element.find_all("td", recursive=False)
            if len(cells) == 2:
                number, prose = _text(cells[0]), _text_outside_tables(cells[1])
                if prose:
                    items.append((None, f"{number} {prose}".strip()))
            continue

        if element.find_parent("tr") or _has_class(element, _NOISE_CLASS):
            continue

        text = _text(element)
        if text:
            items.append((_heading_class(element), text))

    return items


def _section_depths(
    items: list[tuple[str | None, str]], anchor: int | None
) -> dict[str, int]:
    """Dense-rank the heading classes that actually occur in this document.

    The ``Judgment`` anchor is deliberately excluded from the ranking. Including it
    makes ``Legal context`` a *sibling* of ``Judgment`` under the modern scheme but a
    *child* of it under the legacy scheme: the same section landing at a different
    depth, and therefore a different Neo4j parent, depending only on the year the
    judgment was published. Excluding it puts ``Legal context`` at depth 0 in both.
    """
    start = 0 if anchor is None else anchor + 1
    present = {css for css, _ in items[start:] if css}
    ranked = sorted(present, key=_heading_rank)
    return {css: depth for depth, css in enumerate(ranked)}


def _find_anchor(items: list[tuple[str | None, str]]) -> int | None:
    return next(
        (i for i, (css, text) in enumerate(items) if css and text.strip().lower() == _ANCHOR),
        None,
    )


def _find_topics(soup: BeautifulSoup) -> str:
    block = soup.find(class_=_INDEX_CLASS)
    return _text(block) if block else ""


def _split_topics(raw: str) -> list[str]:
    """Split the index block into topics, de-duplicated, in document order.

    A judgment can list the same qualifier twice (62018CJ0511 has "Scope" under both
    Directive 2002/58 and Directive 2000/31). Since the node id is derived from the
    label, the duplicate would silently upsert onto the first, so drop it here instead.
    """
    cleaned = raw.strip().lstrip("(").rstrip(")").strip()

    if cleaned and cleaned[0] in _OPEN_QUOTES:
        cleaned = cleaned[1:]
        if cleaned and cleaned[-1] in _CLOSE_QUOTES:
            cleaned = cleaned[:-1]

    topics = (topic.strip() for topic in _TOPIC_SPLIT.split(cleaned))
    return list(dict.fromkeys(topic for topic in topics if topic))


def _build_preamble(
    builder: TreeBuilder,
    items: list[tuple[str | None, str]],
    anchor: int,
    topics: str,
) -> None:
    """Emit the three synthetic depth-0 sections that precede the judgment body.

    ``kg_builder`` keys its HAS_TOPIC edges off the literal heading "Topics" and skips all
    three when creating paragraphs, so these names are a contract, not a cosmetic choice.
    """
    topics_idx = next((i for i, (_, text) in enumerate(items[:anchor]) if text == topics), anchor)

    builder.open_section("Reports of Cases", 0)
    for _, text in items[:topics_idx]:
        builder.add_body(text)

    builder.open_section("Topics", 0)
    for topic in _split_topics(topics):
        builder.add_body(topic)

    builder.open_section("General Information", 0)
    for _, text in items[topics_idx + 1 : anchor]:
        builder.add_body(text)


def parse_case_law(html: str) -> list[Node]:
    """Parse an EUR-Lex judgment into the same Node tree ``flatten()`` expects."""
    soup = BeautifulSoup(html, "html.parser")
    items = _linearize(soup)
    if not items:
        raise CaseLawHTMLError("Document contains no parsable content.")

    anchor = _find_anchor(items)
    depths = _section_depths(items, anchor)
    if not depths:
        raise CaseLawHTMLError("Document contains no structural headings.")

    builder = TreeBuilder()

    if anchor is None:
        logger.warning("No 'Judgment' anchor found; parsing without a preamble.")
    else:
        _build_preamble(builder, items, anchor, _find_topics(soup))

    start = 0 if anchor is None else anchor
    for css, text in items[start:]:
        if css is None:
            builder.add_body(text)
        elif text.strip().lower() == _ANCHOR:
            builder.open_section(text, 0)
        else:
            builder.open_section(text, depths[css])

    return builder.roots


def parse_celex(celex: str) -> list[Node]:
    """Fetch *celex* from Cellar and parse it. The whole parsing pipeline."""
    roots = parse_case_law(fetch_html(celex))
    logger.info("Parsed %s: %d top-level sections", celex, len(roots))
    return roots
