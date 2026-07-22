"""Prompt library, grouped by the stage each prompt serves.

Every prompt is a versioned :class:`~legal_assistant.rag.prompts.registry.PromptVersion`
registered into the shared ``registry``. The names exported here resolve to the *active*
version's body, so callers import a plain string and never think about versions::

    from legal_assistant.rag.prompts import ANSWER_SYNTHESIS_PROMPT

To ship a new version of a prompt: add a ``<NAME>_V<n>`` text constant in the domain
module, register it with ``active=True``, and flip the previous version to
``active=False``. Rollback is the reverse flip. To A/B a version without touching the
flags, pin it explicitly with ``registry.get("answer_synthesis", "v9")``.

The four domain modules are imported for their registration side effect.
"""
from legal_assistant.rag.prompts import case_law, retrieval, summaries, synthesis  # noqa: F401
from legal_assistant.rag.prompts.registry import PromptRegistry, PromptVersion, registry

# ── active-version exports ────────────────────────────────────────────────────

QUERY_CLASSIFICATION_PROMPT = registry.active("query_classification").body
TOPIC_SELECTION_PROMPT = registry.active("topic_selection").body
HYDE_PROMPT = registry.active("hyde").body

ANSWER_SYNTHESIS_PROMPT = registry.active("answer_synthesis").body
ANSWER_FILTER_PROMPT = registry.active("answer_filter").body
CONTEXT_CURATION_PROMPT = registry.active("context_curation").body
ATTRIBUTION_PROMPT = registry.active("attribution").body

ARTICLE_SUMMARY_SYSTEM_PROMPT = registry.active("article_summary_system").body
ARTICLE_SUMMARY_USER_PROMPT = registry.active("article_summary_user").body
CHAPTER_SUMMARY_SYSTEM_PROMPT = registry.active("chapter_summary_system").body
CHAPTER_SUMMARY_USER_PROMPT = registry.active("chapter_summary_user").body

CASE_LAW_ENTITY_SUMMARY_SYSTEM_PROMPT = registry.active("case_law_entity_summary_system").body
CASE_LAW_ENTITY_SUMMARY_USER_PROMPT = registry.active("case_law_entity_summary_user").body
CASE_LAW_ENTIRE_DOC_SUMMARY_SYSTEM_PROMPT = registry.active("case_law_entire_doc_summary_system").body
CASE_LAW_ENTIRE_DOC_SUMMARY_USER_PROMPT = registry.active("case_law_entire_doc_summary_user").body

__all__ = [
    "registry",
    "PromptRegistry",
    "PromptVersion",
    "QUERY_CLASSIFICATION_PROMPT",
    "TOPIC_SELECTION_PROMPT",
    "HYDE_PROMPT",
    "ANSWER_SYNTHESIS_PROMPT",
    "ANSWER_FILTER_PROMPT",
    "CONTEXT_CURATION_PROMPT",
    "ATTRIBUTION_PROMPT",
    "ARTICLE_SUMMARY_SYSTEM_PROMPT",
    "ARTICLE_SUMMARY_USER_PROMPT",
    "CHAPTER_SUMMARY_SYSTEM_PROMPT",
    "CHAPTER_SUMMARY_USER_PROMPT",
    "CASE_LAW_ENTITY_SUMMARY_SYSTEM_PROMPT",
    "CASE_LAW_ENTITY_SUMMARY_USER_PROMPT",
    "CASE_LAW_ENTIRE_DOC_SUMMARY_SYSTEM_PROMPT",
    "CASE_LAW_ENTIRE_DOC_SUMMARY_USER_PROMPT",
]
