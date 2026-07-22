"""Post-synthesis source attribution.

Splits an already-written prose answer into sentence-level :class:`Segment`\\ s, then
asks an LLM to tag each sentence with the ``[Sn]`` markers of the retrieved sources
that support it.

The answer is split **locally** (NLTK), so the answer text is never altered and
the segments reconstruct it by construction; the LLM only returns marker
assignments keyed by sentence index. This runs as a second pass *after*
synthesis, keeping the synthesis prompt focused on producing clean legal prose.

The three dataclasses below describe the result: an answer broken into atomic claims,
each linked to the exact retrieved passages that support it. They live here rather than
with the retrieval-strategy contract in :mod:`legal_assistant.rag.methods.base`, which
is about *how* passages are found, not how a finished answer is cited.
"""
from __future__ import annotations

import dataclasses
import logging
import re
from dataclasses import dataclass, field
from typing import List, Tuple

import nltk
from langchain_core.language_models import BaseChatModel
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel, Field

from legal_assistant.rag.prompts import ATTRIBUTION_PROMPT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceRef:
    """A single retrieved passage, numbered with a stable marker (e.g. ``S1``)."""

    marker: str
    doc_id: str
    act: str
    title: str
    type: str
    text: str


@dataclass
class Segment:
    """One atomic claim of the answer and the source markers that support it."""

    text: str
    source_markers: List[str] = field(default_factory=list)


@dataclass
class AttributedAnswer:
    """A synthesised answer with per-claim source attribution."""

    segments: List[Segment]
    sources: List[SourceRef]
    raw_answer: str


nltk.download("punkt_tab", quiet=True)

# Fallback splitter for when NLTK data is unavailable: break after sentence-ending
# punctuation that is followed by whitespace and a new clause (capital / opening
# paren / quote), which keeps legal cites like "Article 24(1)." intact.
_SENTENCE_FALLBACK_RE = re.compile(r"(?<=[.;])\s+(?=[A-Z(\"'])")


class _Assignment(BaseModel):
    """The markers supporting one sentence, keyed by its index in the answer."""

    index: int = Field(description="Zero-based index of the sentence.")
    markers: List[str] = Field(
        default_factory=list,
        description="Supporting source markers for this sentence, e.g. ['S1', 'S2'].",
    )


class _AttributionResult(BaseModel):
    """Marker assignments, one per sentence."""

    assignments: List[_Assignment]


def _split_sentences(text: str) -> List[str]:
    """Split ``text`` into trimmed, non-empty sentences (NLTK, regex fallback)."""
    try:
        sentences = sent_tokenize(text)
    except LookupError:  # NLTK punkt data missing and download unavailable
        sentences = _SENTENCE_FALLBACK_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _format_sources(sources: List[SourceRef]) -> str:
    """Render the sources as a numbered block the model can match claims against."""
    blocks = []
    for src in sources:
        header = f"[{src.marker}] {src.act} — {src.title} ({src.doc_id})"
        blocks.append(f"{header}\n{src.text}")
    return "\n\n".join(blocks)


def attribute_answer(
    answer_text: str, sources: List[SourceRef], llm: BaseChatModel
) -> List[Segment]:
    """Split ``answer_text`` into sentences tagged with supporting ``[Sn]`` markers.

    Falls back to a single unmarked segment holding the whole answer when there
    are no sources or the LLM attribution call fails.
    """
    if not sources:
        return [Segment(text=answer_text)]

    sentences = _split_sentences(answer_text)
    if not sentences:
        return [Segment(text=answer_text)]

    valid_markers = {src.marker for src in sources}
    numbered = "\n".join(f"[{i}] {sentence}" for i, sentence in enumerate(sentences))
    prompt = ATTRIBUTION_PROMPT.format(
        sentences=numbered, sources=_format_sources(sources)
    )

    try:
        result = llm.with_structured_output(_AttributionResult).invoke(prompt)
    except Exception:  # external boundary (LLM / parsing) — log and degrade gracefully
        logger.exception("[attribution] attribution call failed; using single segment")
        return [Segment(text=answer_text)]

    assignments = result.assignments if result else []
    markers_by_index = {
        a.index: [m for m in a.markers if m in valid_markers] for a in assignments
    }

    segments = [
        Segment(text=sentence, source_markers=markers_by_index.get(i, []))
        for i, sentence in enumerate(sentences)
    ]
    tagged = sum(1 for seg in segments if seg.source_markers)
    logger.info(
        "[attribution] %d/%d sentence(s) attributed to a source", tagged, len(segments)
    )
    return segments


def keep_cited_sources(
    segments: List[Segment], sources: List[SourceRef]
) -> Tuple[List[Segment], List[SourceRef]]:
    """Drop sources no sentence cites and renumber the rest as S1, S2, … .

    Retrieval is high-recall, so some retrieved passages never make it into the
    answer. This keeps only the cited ones, renumbered consecutively in order of
    first appearance, and rewrites the segment markers to match. If nothing was
    cited (e.g. the attribution call failed), the sources are returned unchanged
    so the user still sees what was retrieved.
    """
    used_order: List[str] = []
    seen = set()
    for seg in segments:
        for marker in seg.source_markers:
            if marker not in seen:
                seen.add(marker)
                used_order.append(marker)

    if not used_order:
        return segments, sources

    remap = {old: f"S{i}" for i, old in enumerate(used_order, 1)}
    src_by_marker = {src.marker: src for src in sources}
    new_sources = [
        dataclasses.replace(src_by_marker[old], marker=new)
        for old, new in remap.items()
    ]
    new_segments = [
        Segment(text=seg.text, source_markers=[remap[m] for m in seg.source_markers])
        for seg in segments
    ]
    return new_segments, new_sources
