"""Read the analysis stage's JSON output into :class:`AnalysedObligation` records.

The stage emits, per passage, a JSON array of obligations. Each obligation names its type and
a predicate, and gives every other element as a list of ``{extraction_method, value}``. This
reader is written against a real committed output, not the paper's listing, so a divergence
surfaces in a test rather than during ingestion.
"""
from __future__ import annotations

from typing import List, Optional

from legal_assistant.obligations.models import (
    AnalysedObligation,
    ExtractedElement,
    ExtractedPredicate,
    ObligationType,
    method_from_text,
    obligation_id,
)

_TYPE_FROM_TEXT = {
    "obligation of action": ObligationType.ACTION,
    "obligation of being": ObligationType.BEING,
}


def _elements(raw: Optional[list]) -> List[ExtractedElement]:
    return [
        ExtractedElement(value=item.get("value"), method=method_from_text(item.get("extraction_method")))
        for item in (raw or [])
    ]


def _predicate(raw: dict) -> ExtractedPredicate:
    return ExtractedPredicate(
        value=raw.get("value", ""),
        verb=raw.get("verb"),
        method=method_from_text(raw.get("extraction_method")),
    )


def _obligation_type(raw: str) -> ObligationType:
    return _TYPE_FROM_TEXT.get((raw or "").strip().lower(), ObligationType.ACTION)


def parse_analysis(
    record: list, source_id: str, celex: str, start_ordinal: int = 1
) -> List[AnalysedObligation]:
    """Parse one analysis response into obligations, numbered from ``start_ordinal``.

    ``record`` is the JSON array the analysis stage produced for a single sentence; ``source_id``
    is the passage's graph id, which every obligation id is composed from. A passage can hold
    several deontic sentences, each analysed on its own, so numbering continues across them:
    ``start_ordinal`` is where this sentence's obligations pick up, keeping ids unique within
    the passage.
    """
    return [
        AnalysedObligation(
            id=obligation_id(source_id, ordinal),
            celex=celex,
            obligation_type=_obligation_type(item.get("ObligationTypeClassification")),
            predicate=_predicate(item.get("Predicate", {})),
            addressees=_elements(item.get("Addressees")),
            targets=_elements(item.get("Targets")),
            specifications=_elements(item.get("Specifications")),
            preconditions=_elements(item.get("Pre-Conditions")),
            beneficiaries=_elements(item.get("Beneficiaries")),
        )
        for ordinal, item in enumerate(record, start=start_ordinal)
    ]
