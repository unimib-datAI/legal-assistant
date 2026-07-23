"""The obligation stages use their own model, kept apart from the RAG pipeline's.

Extraction and vocabulary generation run on a stronger model (gpt-5-mini): on the GDPR
definitions it recovered a real IS_A and two entity subjects gpt-4o-mini missed. Retrieval
stays on gpt-4o-mini. Sharing one setting would couple two decisions that were made
separately, so they are two settings.
"""
from __future__ import annotations

from legal_assistant import config


def test_extraction_model_is_defined():
    assert config.EXTRACTION_LLM_MODEL


def test_extraction_model_is_not_the_rag_model():
    """A single knob would drag retrieval onto the extraction model, or the reverse."""
    assert config.EXTRACTION_LLM_MODEL != config.RAG_LLM_MODEL
