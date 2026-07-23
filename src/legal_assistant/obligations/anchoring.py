"""Anchor extracted addressee and beneficiary strings onto the actor vocabulary.

The extraction names parties in free text: "The provider or its legal representative",
"providers of high-risk AI systems", "Member State". None of these can drive a graph filter
as written. Anchoring turns each into one or more actor ids, extending the vocabulary where
the acts' Definitions articles could not.

Four outcomes, only the last discards:

1. **Match.** A part whose meaning is close enough to an actor is anchored to it.
2. **Containment.** A part that matches nothing but contains a known actor's label becomes a
   qualified child of that actor. This is the hierarchy tier the definitions do not carry.
3. **Frequency.** A part that neither matches nor contains one, but recurs at or above a
   floor, becomes an actor on its own. This is how institutional subjects such as the
   Commission and Member States, defined in no Definitions article, enter the vocabulary.
4. **Unmapped.** Everything else, kept with its count as a diagnostic.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Callable, Dict, List, Sequence

from pydantic import BaseModel, ConfigDict, Field

from legal_assistant.obligations.models import Actor
from legal_assistant.obligations.vocabulary import slugify

logger = logging.getLogger(__name__)

EmbedFn = Callable[[List[str]], List[Sequence[float]]]

# Conjunctions that join co-addressees: "the provider or its representative", "A and B".
_CONJUNCTION = re.compile(r"\s*(?:,|\bor\b|\band\b)\s*", re.IGNORECASE)

# A leading article, dropped before testing an actor label for containment.
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


class AnchorResult(BaseModel):
    """The outcome of anchoring a corpus of strings."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    resolved: Dict[str, List[str]] = Field(default_factory=dict)
    promoted: List[Actor] = Field(default_factory=list)
    unmapped: Counter = Field(default_factory=Counter)


def split_conjunctions(text: str) -> List[str]:
    """Split a co-addressee string into its parts, dropping empties."""
    return [part for raw in _CONJUNCTION.split(text) if (part := raw.strip())]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def _contained_actor(part: str, actors: Sequence[Actor]) -> Actor | None:
    """The actor whose label appears as words inside ``part``, longest label first.

    Longest first so "provider of a general-purpose AI model" prefers the most specific label
    it contains. The label is matched on word boundaries, article-stripped, to avoid a
    substring hit inside an unrelated word.
    """
    words = _LEADING_ARTICLE.sub("", part.lower())
    best = None
    for actor in actors:
        label = _LEADING_ARTICLE.sub("", actor.label.lower())
        if re.search(rf"\b{re.escape(label)}s?\b", words):
            if best is None or len(label) > len(_LEADING_ARTICLE.sub("", best.label.lower())):
                best = actor
    return best


def anchor_strings(
    strings: Counter,
    actors: Sequence[Actor],
    embed: EmbedFn,
    *,
    threshold: float,
    frequency_floor: int,
) -> AnchorResult:
    """Anchor every string in ``strings`` onto ``actors``, extending the vocabulary as needed."""
    parts_of = {text: split_conjunctions(text) for text in strings}
    unique_parts = sorted({p for parts in parts_of.values() for p in parts})

    actor_vectors = dict(zip((a.id for a in actors), embed([a.label for a in actors])))
    part_vectors = dict(zip(unique_parts, embed(unique_parts)))

    known = {a.id: a for a in actors}
    promoted: Dict[str, Actor] = {}
    resolved: Dict[str, List[str]] = {}
    leftover_parts: Counter = Counter()
    # Which strings a leftover part came from, so they can be resolved once it is promoted.
    strings_of_part: Dict[str, set] = {}

    # Pass 1: resolve each part by match or containment; defer the rest.
    part_resolution: Dict[str, List[str]] = {}
    for part in unique_parts:
        matched = _match(part, part_vectors[part], known, actor_vectors, threshold)
        if matched:
            part_resolution[part] = matched
            continue

        container = _contained_actor(part, list(known.values()) + list(promoted.values()))
        if container is not None:
            child = _promote_child(part, container)
            promoted.setdefault(child.id, child)
            part_resolution[part] = [child.id]
            continue

        part_resolution[part] = []  # deferred to the frequency pass

    # Tally deferred parts across all the strings that contain them.
    for text, count in strings.items():
        for part in parts_of[text]:
            if not part_resolution[part]:
                leftover_parts[part] += count
                strings_of_part.setdefault(part, set()).add(text)

    # Pass 2: promote recurring leftovers; the rest are unmapped.
    unmapped: Counter = Counter()
    for part, count in leftover_parts.items():
        if count >= frequency_floor:
            actor = Actor(id=slugify(part), label=part)
            promoted.setdefault(actor.id, actor)
            part_resolution[part] = [actor.id]
        else:
            unmapped[part] += count

    # Assemble per-string resolution from its parts, in order, deduplicated.
    for text, parts in parts_of.items():
        ids: List[str] = []
        for part in parts:
            for actor_id in part_resolution[part]:
                if actor_id not in ids:
                    ids.append(actor_id)
        if ids:
            resolved[text] = ids

    logger.info(
        "[anchoring] %d string(s): %d resolved, %d actor(s) promoted, %d unmapped",
        len(strings), len(resolved), len(promoted), len(unmapped),
    )
    return AnchorResult(resolved=resolved, promoted=list(promoted.values()), unmapped=unmapped)


def _match(
    part: str,
    vector: Sequence[float],
    known: Dict[str, Actor],
    actor_vectors: Dict[str, Sequence[float]],
    threshold: float,
) -> List[str]:
    """Actor ids whose meaning is within ``threshold`` of ``part``, best first."""
    scored = [
        (actor_id, _cosine(vector, actor_vectors[actor_id]))
        for actor_id in known
    ]
    hits = sorted(
        ((actor_id, score) for actor_id, score in scored if score >= threshold),
        key=lambda pair: -pair[1],
    )
    return [actor_id for actor_id, _ in hits]


def _promote_child(part: str, parent: Actor) -> Actor:
    """A qualified child actor: the extracted phrase, qualifying the actor it contains."""
    return Actor(id=slugify(part), label=part.strip(), celex=parent.celex, is_a=[parent.id])
