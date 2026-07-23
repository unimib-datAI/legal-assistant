"""Fetch the text of cited provisions, for the analysis stage's Citations section.

Adapted from the source repository's ``solve_references``. Theirs resolved paragraph ids
inside an in-memory dataset; this resolves the same-act article ids that detection records,
reading their text from the graph. The paper found that giving the analysis stage the content
of cross-referenced provisions improves extraction, because an obligation's elements are often
dispersed across the provisions it cites.
"""
from __future__ import annotations

import logging
from typing import List, Sequence

from legal_assistant.graph.queries import NodeQueries
from legal_assistant.rag.acts import CELEX_TO_ACT_NAME

logger = logging.getLogger(__name__)


def _celex_of(article_id: str) -> str:
    """The owning act's CELEX, the id up to the ``art_`` marker."""
    return article_id.split("art_", 1)[0]


def _header(article_id: str, title: str) -> str:
    act_name = CELEX_TO_ACT_NAME.get(_celex_of(article_id), _celex_of(article_id))
    return f"[{act_name}, Article — {title}]"


def citation_contents(graph, article_ids: Sequence[str]) -> List[str]:
    """The titled text of each cited article, in request order, unknown ids skipped.

    An id the graph does not hold is dropped rather than rendered empty: a citation section
    must not carry a heading with no provision under it.
    """
    if not article_ids:
        return []

    rows = graph.query(NodeQueries.GET_ARTICLES_BY_IDS, params={"ids": list(article_ids)})
    by_id = {row["id"]: row for row in rows}

    contents = []
    for article_id in article_ids:
        row = by_id.get(article_id)
        if row is None:
            continue
        contents.append(f"{_header(article_id, row['title'])}\n{row['text']}".strip())

    if len(contents) < len(article_ids):
        logger.debug("[references] %d/%d cited article(s) not in the graph",
                     len(article_ids) - len(contents), len(article_ids))
    return contents
