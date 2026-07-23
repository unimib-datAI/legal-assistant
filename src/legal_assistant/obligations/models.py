"""Graph-node shapes for obligations and actors.

An obligation is anchored to the passage it was extracted from, never to the article: the
passage id is already this project's citation unit, so attribution and cross-referencing work
unchanged. The obligation id embeds that passage id, which is what lets retrieval recover the
source with a string split instead of a query.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

# Separates the passage id from the obligation's ordinal within that passage.
_ID_MARKER = "#ob_"


class ExtractionMethod(str, Enum):
    """Where an element came from, and therefore how far it can be trusted.

    Declared strongest first. ``NONE`` is not a weak method, it marks an element that is
    simply not present, and so takes no part in the comparison.
    """

    STATED = "STATED"
    CONTEXT = "CONTEXT"
    CITATION = "CITATION"
    BACKGROUND = "BACKGROUND"
    NONE = "NONE"


# Strongest to weakest. Index order is the comparison; NONE is excluded by construction.
_METHOD_STRENGTH: tuple[ExtractionMethod, ...] = (
    ExtractionMethod.STATED,
    ExtractionMethod.CONTEXT,
    ExtractionMethod.CITATION,
    ExtractionMethod.BACKGROUND,
)


# The analysis stage writes extraction methods in this surface form; map to the enum.
_METHOD_FROM_TEXT = {
    "stated": ExtractionMethod.STATED,
    "context": ExtractionMethod.CONTEXT,
    "citation": ExtractionMethod.CITATION,
    "background-knowledge": ExtractionMethod.BACKGROUND,
    "background knowledge": ExtractionMethod.BACKGROUND,
    "none": ExtractionMethod.NONE,
}


def weakest_of(methods) -> ExtractionMethod:
    """The least reliable populated method in a collection, NONE if none is populated.

    Reduces a multi-valued element (several addressees, several targets) to the single method
    that governs how far the whole can be trusted.
    """
    populated = [m for m in methods if m is not ExtractionMethod.NONE]
    if not populated:
        return ExtractionMethod.NONE
    return max(populated, key=_METHOD_STRENGTH.index)


def method_from_text(value: str) -> ExtractionMethod:
    """Parse an extraction method as the analysis JSON writes it.

    Tolerant of case and of the "Background-Knowledge" / "Background Knowledge" spelling. An
    unknown value is treated as NONE rather than raising: a malformed method must not lose the
    obligation it belongs to.
    """
    return _METHOD_FROM_TEXT.get((value or "").strip().lower(), ExtractionMethod.NONE)


class ExtractedElement(BaseModel):
    """One value of one obligation element, with where it was drawn from.

    Elements are multi-valued in the source output: an obligation may have several addressees
    or beneficiaries, so each is a list of these.
    """

    value: Optional[str] = None
    method: ExtractionMethod = ExtractionMethod.NONE


class ExtractedPredicate(BaseModel):
    """The predicate, the one mandatory element, carrying its voice."""

    value: str
    verb: Optional[str] = None
    method: ExtractionMethod = ExtractionMethod.NONE


class Modality(str, Enum):
    """A prohibition is the obligation to refrain, so both are deontic obligations."""

    OBLIGATION = "OBLIGATION"
    PROHIBITION = "PROHIBITION"


class ObligationType(str, Enum):
    """An obligation of action requires doing; an obligation of being requires a state."""

    ACTION = "ACTION"
    BEING = "BEING"


class AnalysedObligation(BaseModel):
    """One obligation as the analysis stage structures it, before it becomes graph nodes.

    Faithful to the source output shape: lists of :class:`ExtractedElement` for every element
    but the predicate. The graph ``Obligation`` and its edges are derived from this at
    ingestion; keeping the two apart means the reader is written against reality and the node
    mapping is a separate, testable step.
    """

    id: str
    celex: str
    obligation_type: ObligationType
    predicate: ExtractedPredicate
    addressees: List[ExtractedElement] = Field(default_factory=list)
    targets: List[ExtractedElement] = Field(default_factory=list)
    specifications: List[ExtractedElement] = Field(default_factory=list)
    preconditions: List[ExtractedElement] = Field(default_factory=list)
    beneficiaries: List[ExtractedElement] = Field(default_factory=list)


def source_id_of(obligation_id: str) -> str:
    """The passage an obligation was extracted from, read off its id.

    Retrieval fuses an obligation's source passage back into the article ranking, and doing
    that per obligation would otherwise cost a query each.
    """
    passage_id, marker, _ = obligation_id.partition(_ID_MARKER)
    if not marker:
        raise ValueError(f"{obligation_id!r} is not an obligation id: no {_ID_MARKER!r}")
    return passage_id


def obligation_id(source_id: str, ordinal: int) -> str:
    """Compose the id of the ``ordinal``-th obligation extracted from ``source_id``."""
    return f"{source_id}{_ID_MARKER}{ordinal}"


class Actor(BaseModel):
    """A subject the legislation names and defines, and to which obligations are addressed.

    Not a discovered cluster: the vocabulary is generated from the acts' own Definitions
    articles, so an actor is a subject the legislature itself chose to define.
    """

    id: str = Field(description="Slug, stable across regenerations, e.g. 'provider'.")
    label: str = Field(description="Human-readable form, e.g. 'Provider'.")
    celex: Optional[str] = Field(
        default=None,
        description="Act that defines it; None for subjects bearing duties under several.",
    )
    defined_in: Optional[str] = Field(
        default=None, description="Id of the paragraph defining it, when there is one."
    )
    is_a: List[str] = Field(
        default_factory=list,
        description="Actors this one qualifies. A provider of a high-risk AI system is a "
                    "provider, so a filter on 'provider' must reach it.",
    )


class Obligation(BaseModel):
    """One deontic obligation, as the analysis stage structures it.

    Only the predicate is mandatory. Every other element is optional in the source framework,
    including the addressee, which legal texts frequently leave implicit.
    """

    id: str
    celex: str
    modality: Modality
    obligation_type: ObligationType

    predicate_text: str
    predicate_voice: Optional[str] = None
    target: Optional[str] = None
    specification: Optional[str] = None
    precondition: Optional[str] = None
    beneficiary_text: Optional[str] = None

    addressee_method: ExtractionMethod = ExtractionMethod.NONE
    predicate_method: ExtractionMethod = ExtractionMethod.NONE
    target_method: ExtractionMethod = ExtractionMethod.NONE
    specification_method: ExtractionMethod = ExtractionMethod.NONE
    precondition_method: ExtractionMethod = ExtractionMethod.NONE
    beneficiary_method: ExtractionMethod = ExtractionMethod.NONE

    @property
    def source_id(self) -> str:
        """The passage this obligation was extracted from."""
        return source_id_of(self.id)

    @property
    def weakest_method(self) -> ExtractionMethod:
        """The least reliable element actually present.

        An answer resting on this obligation is only as defensible as its weakest part, so a
        single element the model inferred from background knowledge governs how the whole is
        presented. Absent elements are excluded: not being there is not the same as being
        badly extracted.
        """
        populated = [
            method for method in (
                self.addressee_method,
                self.predicate_method,
                self.target_method,
                self.specification_method,
                self.precondition_method,
                self.beneficiary_method,
            )
            if method is not ExtractionMethod.NONE
        ]
        if not populated:
            return ExtractionMethod.NONE
        return max(populated, key=_METHOD_STRENGTH.index)
