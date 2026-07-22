"""Prompts for summarising CJEU judgments — per section and whole document."""
from datetime import date

from legal_assistant.rag.prompts.registry import PromptVersion, registry


CASE_LAW_ENTITY_SUMMARY_SYSTEM_V1 = """
You are an AI assistant that helps a human analyst to perform general information discovery. Information
discovery is the process of identifying and assessing relevant information associated with certain
entities (e.g., organizations and individuals) within a network.
"""

CASE_LAW_ENTITY_SUMMARY_USER_V1 = """
You are analyzing a single section of a structured EU legal document.

Section heading: {heading}
Section depth: {depth}
Section body:
{body}

Produce a concise summary (3–6 sentences) of this section capturing:
1. The main legal question or topic addressed
2. The key parties, authorities, or legal instruments mentioned
3. The core finding, ruling, or argument made
4. Any notable references to other sections or legal precedents (if present)

Return ONLY a JSON object with the following fields:
- "heading": the section heading (copied exactly as given)
- "summary": your concise summary
"""

CASE_LAW_ENTIRE_DOC_SUMMARY_SYSTEM_V1 = """You are an expert legal document summarizer."""

CASE_LAW_ENTIRE_DOC_SUMMARY_USER_V1 = """
Summarize the following CJEU judgment. Focus on:
- The case number (e.g., C-XXX/YY)
- The parties
- The specific articles or regulations interpreted
- The core legal question

The summary must be concise, maximum {char_length}
characters long, and optimized for providing context
to smaller text chunks. Output only the summary text.

Document: {document_content}
"""

registry.register(PromptVersion(
    name="case_law_entity_summary_system", version="v1", created=date(2026, 6, 20),
    notes="Initial tracked version. System prompt for section-level entity "
          "summaries.",
    body=CASE_LAW_ENTITY_SUMMARY_SYSTEM_V1, active=True,
))

registry.register(PromptVersion(
    name="case_law_entity_summary_user", version="v1", created=date(2026, 6, 20),
    notes="Initial tracked version. User prompt for summarising a single "
          "document section as JSON.",
    body=CASE_LAW_ENTITY_SUMMARY_USER_V1, active=True,
))

registry.register(PromptVersion(
    name="case_law_entire_doc_summary_system", version="v1", created=date(2026, 6, 20),
    notes="Initial tracked version. System prompt for whole-judgment "
          "summarisation.",
    body=CASE_LAW_ENTIRE_DOC_SUMMARY_SYSTEM_V1, active=True,
))

registry.register(PromptVersion(
    name="case_law_entire_doc_summary_user", version="v1", created=date(2026, 6, 20),
    notes="Initial tracked version. User prompt for summarising an entire "
          "CJEU judgment within a character budget.",
    body=CASE_LAW_ENTIRE_DOC_SUMMARY_USER_V1, active=True,
))
