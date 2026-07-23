"""Generate a compliance checklist: every obligation an actor bears, rendered by the LLM.

Deliberately not routed through retrieval. The set is fetched by a deterministic Cypher query
that walks the actor hierarchy, so it is complete by construction; the LLM only renders it. A
top-k retriever would silently truncate exactly where completeness is the point.
"""
from __future__ import annotations

import logging
from typing import List

from langchain_core.messages import HumanMessage

from legal_assistant.graph.queries import NodeQueries
from legal_assistant.rag.acts import CELEX_TO_ACT_NAME
from legal_assistant.rag.prompts import OBLIGATION_CHECKLIST_PROMPT

logger = logging.getLogger(__name__)


def fetch_checklist(graph, celex: str, actor: str) -> List[dict]:
    """Every obligation addressed to ``actor`` or a more specific actor, ordered by provision.

    The ``IS_A*0..`` walk is what makes a checklist for "provider" include the obligations of
    "provider of a high-risk AI system": the child is_a the parent, so filtering on the parent
    reaches every child.
    """
    return graph.query(
        NodeQueries.GET_OBLIGATIONS_FOR_ACTOR,
        params={"celex": celex, "actor": actor},
    )


def _render_obligation(row: dict) -> str:
    parts = [f"- [{row['source_id']}] ({row['modality']}, {row['obligation_type']}) "
             f"{row['predicate_text']}"]
    if row.get("target"):
        parts.append(f"target: {row['target']}")
    if row.get("specification"):
        parts.append(f"specification: {row['specification']}")
    if row.get("precondition"):
        parts.append(f"precondition: {row['precondition']}")
    parts.append(f"trust: {row['weakest_method']}")
    return "; ".join(parts)


def checklist(graph, llm, celex: str, actor: str) -> str:
    """The rendered compliance checklist for ``actor`` under ``celex``.

    Empty when the actor bears no obligations: an empty checklist, not an LLM call on nothing.
    """
    rows = fetch_checklist(graph, celex, actor)
    if not rows:
        logger.info("[checklist] no obligations for actor %r under %s", actor, celex)
        return ""

    obligations = "\n".join(_render_obligation(row) for row in rows)
    act_name = CELEX_TO_ACT_NAME.get(celex, celex)
    prompt = OBLIGATION_CHECKLIST_PROMPT.format(
        actor=f"{actor} (under {act_name})", obligations=obligations)

    logger.info("[checklist] rendering %d obligation(s) for actor %r under %s",
                len(rows), actor, celex)
    return llm.invoke([HumanMessage(content=prompt)]).content
