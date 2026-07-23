"""Detect which graph passages potentially carry a deontic obligation.

This replaces the source repository's ``obligation_detection`` stage. Theirs fetches EUR-Lex
by URL and segments on ``div id`` matching ``^\\d+\\.\\d+``; that misses the 14 AI Act
articles and 17 GDPR articles laid out without numbered paragraph divs, Article 16 among
them, takes the six colliding amending-article ids at face value, and cannot see annex
points. Reading passages from the graph avoids all three, because the loader already
synthesised ``.0`` paragraphs, re-derived the colliding ids, and created ``AnnexPoint``
nodes.

What is kept from the source stage is its shape: split each passage into sentences, keep the
ones carrying a deontic marker, and record the provisions each one cites. The output feeds
the filtering stage unchanged.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

from legal_assistant.graph.queries import NodeQueries

logger = logging.getLogger(__name__)

# A same-act article reference: "Article 11", "Articles 8 and 9", "Article 61(2)". Within an
# act's own text an unqualified "Article N" means article N of that same act; a reference that
# names another instrument ("Article 11 of Regulation (EU) 2016/679") is a cross-act citation,
# resolved in the analysis stage where the other act's CELEX is known, not here.
_SAME_ACT_REF = re.compile(r"\bArticles?\s+(\d+(?:\s*\([^)]*\))?(?:\s*(?:,|and)\s*\d+)*)")
_CROSS_ACT_TAIL = re.compile(r"^\s+of\s+(?:the\s+)?(?:Regulation|Directive|Decision)\b", re.I)
_ARTICLE_NUMBER = re.compile(r"\d+")

# Deontic markers, following the source paper's set with one deliberate narrowing. The paper
# lists "shall", "must", "should", "has" and "have to"; a bare "has"/"have" is an auxiliary
# that matches almost every clause ("the Commission has adopted ..."), so only the "has to" /
# "have to" duty form is kept. Recall lost here is small and the filtering stage is the
# precision step regardless.
_MARKERS = ("shall", "must", "should", "ought to", "has to", "have to")
_MARKER_RE = re.compile(
    r"\b(?:" + "|".join(m.replace(" ", r"\s+") for m in _MARKERS) + r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CandidateSentence:
    """One sentence carrying a deontic marker, with the provisions it cites."""

    sentence: str
    references: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PotentialObligation:
    """A passage that carries at least one candidate sentence, ready for filtering."""

    par_id: str
    celex: str
    text: str
    candidates: List[CandidateSentence]


def _split_sentences(text: str) -> List[str]:
    """Split a passage into sentences.

    Uses NLTK when its data is present, as the rest of the project does, and falls back to a
    simple boundary split so detection never depends on a model download at import time.
    """
    try:
        from nltk.tokenize import sent_tokenize

        return sent_tokenize(text)
    except (ImportError, LookupError):
        return [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if s.strip()]


def _same_act_references(sentence: str, celex: str) -> List[str]:
    """Ids of the same-act articles a sentence cites, in first-appearance order.

    A reference immediately qualified by another instrument is skipped: it cites that act, not
    this one, and belongs to the cross-act resolution the analysis stage does.
    """
    found: List[str] = []
    for match in _SAME_ACT_REF.finditer(sentence):
        if _CROSS_ACT_TAIL.match(sentence[match.end():]):
            continue
        for number in _ARTICLE_NUMBER.findall(match.group(1)):
            article_id = f"{celex}art_{int(number)}"
            if article_id not in found:
                found.append(article_id)
    return found


def detect_in_passage(par_id: str, text: str, celex: str) -> PotentialObligation:
    """Find the candidate obligation sentences in one passage.

    Pure over its inputs, so the marker and citation logic is testable without a graph.
    """
    candidates = [
        CandidateSentence(
            sentence=sentence,
            references=_same_act_references(sentence, celex),
        )
        for sentence in _split_sentences(text)
        if _MARKER_RE.search(sentence)
    ]
    return PotentialObligation(par_id=par_id, celex=celex, text=text, candidates=candidates)


def detect(graph, acts: List[str]) -> List[PotentialObligation]:
    """Every passage of the given acts that carries at least one candidate sentence.

    Passages are the ``Paragraph`` and ``AnnexPoint`` nodes of the acts, never recitals:
    recitals do not bind, and selecting only these node labels is what enforces that.
    """
    rows = (
        graph.query(NodeQueries.GET_PARAGRAPHS_BY_ACTS, params={"acts": acts})
        + graph.query(NodeQueries.GET_ANNEX_POINTS_BY_ACTS, params={"acts": acts})
    )

    detected = [
        potential
        for row in rows
        if (potential := detect_in_passage(row["id"], row["text"], row["celex"])).candidates
    ]
    logger.info(
        "[detection] %d/%d passage(s) carry a candidate obligation across acts %s",
        len(detected), len(rows), acts,
    )
    return detected
