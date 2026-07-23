"""Deontic obligation filtering: keep only the sentences that state a duty.

Not every "shall" is deontic; the paper's filtering prompt sorts a candidate sentence into
one of seven categories, and only two of them, obligation and prohibition, advance to
analysis. This adapter drives the paper's filtering prompt, registered in
`rag/prompts/obligations.py`, with the project's LLM client. The prompt is used unchanged.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from legal_assistant.obligations.json_io import extract_json
from legal_assistant.obligations.prompts import filtering_system_prompt, filtering_user_prompt
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FilteringCategory(str, Enum):
    """The seven categories the filtering prompt assigns.

    Values are the exact strings the prompt uses, so a returned classification maps straight
    through. Anything else is treated as NOT_APPLICABLE.
    """

    DEFINITION = "Definition"
    CONSTITUTIVE_STATEMENT = "Constitutive statement"
    DEONTIC_OBLIGATION = "Deontic obligation"
    ENTITLEMENTS = "Entitlements"
    AUTHORISATIONS = "Authorisations"
    DEONTIC_PROHIBITION = "Deontic prohibition"
    NOT_APPLICABLE = "Not applicable"


# Only these advance to analysis, following the paper's SHALL_TYPES_OF_INTEREST.
_DEONTIC = frozenset({FilteringCategory.DEONTIC_OBLIGATION, FilteringCategory.DEONTIC_PROHIBITION})


class FilteringResult(BaseModel):
    classification: FilteringCategory
    justification: str = ""


def is_deontic(category: FilteringCategory) -> bool:
    """Whether a category carries a duty and so advances to analysis."""
    return category in _DEONTIC


def _category(value: object) -> FilteringCategory:
    """Map a returned classification, tolerating case and unknown labels."""
    text = str(value or "").strip().lower()
    for category in FilteringCategory:
        if category.value.lower() == text:
            return category
    return FilteringCategory.NOT_APPLICABLE


def filter_sentence(
    llm, sentence: str, context: str, citations: Optional[List[str]] = None
) -> FilteringResult:
    """Classify one candidate sentence with the paper's filtering prompt."""
    messages = [
        SystemMessage(content=filtering_system_prompt()),
        HumanMessage(content=filtering_user_prompt(sentence, context, citations)),
    ]
    raw = llm.invoke(messages).content
    parsed = extract_json(raw)

    if not isinstance(parsed, dict):
        logger.warning("[filtering] unparseable response for %r; treating as Not applicable",
                       sentence[:60])
        return FilteringResult(classification=FilteringCategory.NOT_APPLICABLE)

    return FilteringResult(
        classification=_category(parsed.get("classification")),
        justification=str(parsed.get("justification", "")),
    )
