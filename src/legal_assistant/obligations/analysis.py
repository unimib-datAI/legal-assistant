"""Deontic obligation analysis: extract the structured elements of one obligation.

The paper's analysis prompt turns a deontic sentence into a JSON array of obligations, each
with its type, predicate, addressees, targets, specifications, pre-conditions and
beneficiaries, every element paired with the method it was extracted by. This adapter drives
the paper's analysis prompt, registered in `rag/prompts/obligations.py`, and hands its output
to ``parse_analysis``, which is pinned against a real output.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from legal_assistant.obligations.extraction_io import parse_analysis
from legal_assistant.obligations.json_io import extract_json
from legal_assistant.obligations.models import AnalysedObligation
from legal_assistant.obligations.prompts import analysis_system_prompt, analysis_user_prompt

logger = logging.getLogger(__name__)


def analyse_sentence(
    llm,
    source_id: str,
    celex: str,
    sentence: str,
    context: str,
    citations: Optional[List[str]] = None,
    start_ordinal: int = 1,
) -> List[AnalysedObligation]:
    """Analyse one deontic sentence into structured obligations.

    ``start_ordinal`` is where this sentence's obligation ids pick up within its passage, so a
    paragraph with several duties does not number two of them alike.

    A response that does not parse into the expected array yields no obligations rather than
    raising: one bad sentence must not stop a batch of hundreds.
    """
    messages = [
        SystemMessage(content=analysis_system_prompt()),
        HumanMessage(content=analysis_user_prompt(sentence, context, citations)),
    ]
    parsed = extract_json(llm.invoke(messages).content)

    if not isinstance(parsed, list):
        logger.warning("[analysis] response was not a JSON array for %r; skipping",
                       sentence[:60])
        return []

    try:
        return parse_analysis(parsed, source_id=source_id, celex=celex,
                              start_ordinal=start_ordinal)
    except (ValidationError, KeyError, TypeError, AttributeError) as exc:
        logger.warning("[analysis] could not read obligations for %r: %s", sentence[:60], exc)
        return []
