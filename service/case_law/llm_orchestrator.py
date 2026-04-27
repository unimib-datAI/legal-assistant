import json
import logging
import re

import config
from service.case_law.doc_parser import extract_sample, build_tree, Node
from service.rag.prompt import (CASE_LAW_DOCUMENT_PARSING_SYSTEM_PROMPT, CASE_LAW_DOCUMENT_PARSING_USER_PROMPT,
                                CASE_LAW_ENTITY_SUMMARY_SYSTEM_PROMPT, CASE_LAW_ENTITY_SUMMARY_USER_PROMPT)

logger = logging.getLogger(__name__)


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

def parse_document(pdf_path: str) -> tuple[dict, list[Node]]:
    """Full pipeline: infer rules then build the document tree. Returns (parsing_rules, roots)."""
    parsing_rules = _infer_config(pdf_path)
    roots = build_tree(pdf_path, parsing_rules)
    return parsing_rules, roots


def summarize_section(section: dict) -> dict:
    """Call the LLM to produce a structured summary for a single flat section."""
    body_text = "\n".join(section.get("body", [])) or "(no direct body text)"
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


def _infer_config(pdf_path: str) -> dict:
    """Extract a structural sample from *pdf_path* and ask the LLM to infer parsing rules:
    1. Extract a sample of text elements with their labels and docling levels which allow the LLM to understand the document's structure.
    2. Format this sample into a prompt and call the LLM to infer structural rules in JSON format.
    """
    logger.info("Extracting structural sample from %s", pdf_path)

    sample = extract_sample(pdf_path)
    sample_text = "\n".join(
        f"[{i+1}] label={s['label']}, docling_level={s['docling_level']}, text=\"{s['text']}\""
        for i, s in enumerate(sample)
    )
    user_prompt = CASE_LAW_DOCUMENT_PARSING_USER_PROMPT.replace("{sample}", sample_text)
    logger.info("Calling LLM for rule inference (%d elements in sample)", len(sample))
    llm_raw_response = _call_llm(user_prompt=user_prompt, system_prompt=CASE_LAW_DOCUMENT_PARSING_SYSTEM_PROMPT)
    parsing_rules = _parse_json(llm_raw_response)
    logger.info("Domain inferred: %s", parsing_rules.get("domain", "—"))
    return parsing_rules
