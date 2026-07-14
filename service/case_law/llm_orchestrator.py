import json
import logging
import re

import config
from service.case_law.html_parser import parse_celex
from service.case_law.tree import Node, flatten
from service.rag.prompt import (CASE_LAW_ENTITY_SUMMARY_SYSTEM_PROMPT, CASE_LAW_ENTITY_SUMMARY_USER_PROMPT,
                                CASE_LAW_ENTIRE_DOC_SUMMARY_SYSTEM_PROMPT, CASE_LAW_ENTIRE_DOC_SUMMARY_USER_PROMPT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Section that isn't necessary to summarize because there is no advantage in doing so, e.g. they don't contain substantive content, or they are too generic.
_SKIP_SECTIONS = frozenset({"Reports of Cases", "Topics", "General Information"})


def _call_llm(user_prompt: str, system_prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL or None)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()

def parse_document(celex: str) -> list[Node]:
    """Parse the judgment identified by *celex* into a section tree.

    Structure comes from the EUR-Lex XHTML markup, not from an LLM — see
    ``service/case_law/html_parser.py``.
    """
    return parse_celex(celex)

def summarize_document(roots: list[Node], char_length: int = 150) -> str:
    """Produce a high-level summary of the entire document from the parsed tree."""
    document_content = "\n".join(
        "\n".join([section["heading"], *section["body"]])
        for section in flatten(roots)
    )
    user_prompt = CASE_LAW_ENTIRE_DOC_SUMMARY_USER_PROMPT.format(
        char_length=char_length,
        document_content=document_content,
    )
    logger.info("Summarising full document (%d chars)", len(document_content))
    return _call_llm(user_prompt=user_prompt, system_prompt=CASE_LAW_ENTIRE_DOC_SUMMARY_SYSTEM_PROMPT)

def summarize_section(section: dict) -> dict | None:
    body_text = "\n".join(section.get("body", []))
    if not body_text.strip() or section.get("heading") in _SKIP_SECTIONS:
        return None
    user_prompt = CASE_LAW_ENTITY_SUMMARY_USER_PROMPT.format(
        heading=section["heading"],
        depth=section["depth"],
        body=body_text,
    )
    logger.info("Summarising section: %s", section["heading"])
    raw = _call_llm(user_prompt=user_prompt, system_prompt=CASE_LAW_ENTITY_SUMMARY_SYSTEM_PROMPT)
    try:
        return _parse_json(raw)
    except (json.JSONDecodeError, ValueError):
        return {"heading": section["heading"], "summary": raw}

def create_case_law_kg(
    celex: str,
    flat: list[dict],
    summaries: list[dict],
    graph,
) -> None:
    """Build the case law subgraph in Neo4j from parsed sections and their summaries.

    Graph structure:
      (CaseLaw {celex}) -[:HAS_SECTION]-> (CaseLawSection)  — depth-based parent/child via [:CONTAINS]
      (CaseLaw)         -[:HAS_TOPIC]->   (CaseLawTopic)    — one node per topic string
    """
    summary_by_heading: dict[str, str] = {s["heading"]: s.get("summary", "") for s in summaries}

    case_law_id = f"{celex}"
    doc_summary = summary_by_heading.get("Document Summary", "")
    graph.upsert_graph_node("CaseLaw", {"id": case_law_id, "celex": celex, "summary": doc_summary})
    logger.info("Upserted CaseLaw node: %s", celex)

    topics_section = next((s for s in flat if s["heading"] == "Topics"), None)
    if topics_section:
        for topic in topics_section.get("body", []):
            topic = topic.strip()
            if not topic:
                continue
            topic_id = f"case_law_topic:{celex}:{topic}"
            graph.upsert_graph_node("CaseLawTopic", {"id": topic_id, "label": topic, "celex": celex})
            graph.create_relationship("CaseLaw", "CaseLawTopic", case_law_id, topic_id, "HAS_TOPIC")

    # Stack maps depth → section_id of the most recent ancestor at that depth.
    depth_stack: dict[int, str] = {}

    sections = [s for s in flat if s["heading"] != "Topics"]
    for i, section in enumerate(sections):
        heading = section["heading"]
        depth = section["depth"]
        body = "\n".join(section.get("body", []))
        summary = summary_by_heading.get(heading, "")

        section_id = f"case_law_section:{celex}:{i}:{heading[:60]}"
        graph.upsert_graph_node("CaseLawSection", {
            "id": section_id,
            "heading": heading,
            "depth": depth,
            "body": body,
            "summary": summary,
        })

        # Find the closest ancestor: walk up from depth-1 until we find one in the stack
        parent_id = next(
            (depth_stack[d] for d in range(depth - 1, -1, -1) if d in depth_stack),
            None,
        )
        if parent_id is None:
            graph.create_relationship("CaseLaw", "CaseLawSection", case_law_id, section_id, "HAS_SECTION")
        else:
            parent_label = "CaseLaw" if parent_id == case_law_id else "CaseLawSection"
            graph.create_relationship(parent_label, "CaseLawSection", parent_id, section_id, "CONTAINS")

        depth_stack[depth] = section_id
        # Invalidate any deeper entries so they don't become false ancestors
        for d in list(depth_stack):
            if d > depth:
                del depth_stack[d]

    logger.info("Case law KG created for %s (%d sections)", celex, len(sections))


def _parse_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
        return json.loads(fixed)
