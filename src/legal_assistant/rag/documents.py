"""Shared helpers for the ``Document`` objects that flow through retrieval.

Every retriever produces LangChain ``Document``s that later stages (fusion, reranking,
synthesis, attribution) identify by id and read by their bracketed source header. Those
two conventions are defined here so a new retrieval strategy inherits them for free
instead of re-deriving ids and headers of its own.

The ``decorate_*`` functions each return a **copy**: the corpus caches inside a retriever
hand back the objects they hold, so enriching in place would corrupt the shared corpus.
"""
from __future__ import annotations

import re

from langchain_core.documents import Document

from legal_assistant.rag.acts import CELEX_TO_ACT_NAME

# Leading display number on a recital, e.g. "(46) Whereas …".
_DISPLAY_NUM_RE = re.compile(r"^\((\d+)\)\s*")

# '\nid: <value>' inside page_content produced by Neo4jVector.from_existing_graph.
_PC_ID_RE = re.compile(r"\nid:\s*([^\n]+)")

# Judgment paragraph ids are sequential by construction ("62018CJ0311_par_199"), so a
# passage's neighbours are reachable by arithmetic — no query needed.
_CASE_LAW_PAR_ID_RE = re.compile(r"^(?P<prefix>.+)_par_(?P<number>\d+)$")


def doc_id(doc: Document) -> str:
    """The passage id: from metadata (sparse docs) or parsed out of page_content (dense)."""
    mid = doc.metadata.get("id")
    if mid:
        return mid
    m = _PC_ID_RE.search(doc.page_content)
    return m.group(1).strip() if m else doc.page_content[:80]


def copy_doc(doc: Document) -> Document:
    """Detach a Document from the corpus cache so downstream stages can enrich it freely.

    BM25 hands back the very objects held by the retriever's article/recital caches;
    without this copy, per-query enrichment would write straight into the shared corpus.
    """
    return Document(page_content=doc.page_content, metadata=dict(doc.metadata))


def neighbour_ids(passage_id: str, window: int) -> list[str]:
    """Ids of the paragraphs reading around ``passage_id``: one before, ``window`` after.

    Asymmetric because a judgment is a sequential argument whose conclusion follows its
    reasoning. Operative parts ("_op_N") get nothing: they are a list of holdings, not a
    reading order, so their neighbours carry no argumentative continuity.
    """
    m = _CASE_LAW_PAR_ID_RE.match(passage_id)
    if not m or window < 1:
        return []
    prefix, number = m.group("prefix"), int(m.group("number"))
    return [
        f"{prefix}_par_{i}"
        for i in range(number - 1, number + window + 1)
        if i >= 1 and i != number
    ]


def recital_header(celex: str, recital_text: str) -> str:
    """'[GDPR, Recital 46]' — the display number is read off the pristine recital text."""
    act_name = CELEX_TO_ACT_NAME.get(celex, celex)
    m = _DISPLAY_NUM_RE.match(recital_text)
    if m:
        return f"[{act_name}, Recital {m.group(1)}]"
    return f"[{act_name}, Recital]"


def decorate_article(doc: Document) -> Document:
    """Return a copy of ``doc`` with an act + chapter + title header prepended.

    The chapter lets the synthesis LLM resolve questions scoped to a specific Chapter
    (e.g. "the personal scope of Chapter II") to the right provision.
    """
    act_name = CELEX_TO_ACT_NAME.get(
        doc.metadata.get("act", ""), doc.metadata.get("act", "")
    )
    title = doc.metadata.get("title", "")
    chapter = doc.metadata.get("chapter_number")
    if chapter:
        chapter_title = doc.metadata.get("chapter_title") or ""
        header = f"[{act_name}, Chapter {chapter} — {chapter_title}, {title}]"
    else:
        header = f"[{act_name}, {title}]"
    return Document(page_content=f"{header}\n{doc.page_content}", metadata=dict(doc.metadata))


def decorate_recital(doc: Document) -> Document:
    """Return a copy of ``doc`` with its recital header prepended, display number folded in.

    Reads the display number off the pristine text, so it must run on an undecorated doc.
    """
    header = recital_header(doc.metadata["celex"], doc.page_content)
    body = _DISPLAY_NUM_RE.sub("", doc.page_content)
    return Document(page_content=f"{header}\n{body}", metadata=dict(doc.metadata))


def decorate_case_law(doc: Document) -> Document:
    """Return a copy of ``doc`` headed with the case number, section and paragraph.

    The header is what tells the synthesis LLM this passage is a *ruling about* a provision
    rather than the provision itself — e.g. ``[C-645/19, The first question, para. 71]``.
    """
    meta = doc.metadata
    case = meta.get("case_number") or meta.get("celex", "")
    section = meta.get("section_heading") or ""
    if meta.get("is_operative"):
        locator = f"operative ruling {meta['number']}" if meta.get("number") else "operative part"
    else:
        locator = f"para. {meta['number']}" if meta.get("number") else "judgment"

    header = f"[{case}, {section}, {locator}]" if section else f"[{case}, {locator}]"
    return Document(page_content=f"{header}\n{doc.page_content}", metadata=dict(meta))
