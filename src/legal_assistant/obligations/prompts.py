"""Fill the obligation prompts' placeholders.

The prompt text lives in the versioned registry (`rag/prompts/obligations.py`); this module
supplies the three placeholders the user templates carry, nothing more. The wording that
carries the reported accuracy is not touched.
"""
from __future__ import annotations

from typing import List, Optional

from legal_assistant.rag.prompts import (
    OBLIGATION_ANALYSIS_SYSTEM_PROMPT,
    OBLIGATION_ANALYSIS_USER_PROMPT,
    OBLIGATION_FILTERING_SYSTEM_PROMPT,
    OBLIGATION_FILTERING_USER_PROMPT,
)

_NO_CITATION = "No Citation"


def filtering_system_prompt() -> str:
    return OBLIGATION_FILTERING_SYSTEM_PROMPT


def analysis_system_prompt() -> str:
    return OBLIGATION_ANALYSIS_SYSTEM_PROMPT


def _fill_user(template: str, sentence: str, context: str, citations: Optional[List[str]]) -> str:
    """Substitute the user template's three placeholders.

    The citations slot is always present; an empty list renders as "No Citation" rather than
    a blank, matching the source behaviour.
    """
    citation_text = "\n".join(citations).strip() if citations else _NO_CITATION
    return (
        template
        .replace("@Sentence", sentence.strip())
        .replace("@Context", (context or "").strip())
        .replace("@Citations", citation_text)
    )


def filtering_user_prompt(sentence: str, context: str, citations: Optional[List[str]] = None) -> str:
    return _fill_user(OBLIGATION_FILTERING_USER_PROMPT, sentence, context, citations)


def analysis_user_prompt(sentence: str, context: str, citations: Optional[List[str]] = None) -> str:
    return _fill_user(OBLIGATION_ANALYSIS_USER_PROMPT, sentence, context, citations)
