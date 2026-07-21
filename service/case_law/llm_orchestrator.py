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

def _parse_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
        return json.loads(fixed)
