import json
import logging
import re

import config
from service.case_law.doc_parser import extract_sample, build_tree, Node
from service.rag.prompt import CASE_LAW_SYSTEM_PROMPT, CASE_LAW_USER_PROMPT

logger = logging.getLogger(__name__)


def _call_llm(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL or None)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": CASE_LAW_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def _parse_json(raw: str) -> dict:
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
        return json.loads(fixed)


def infer_config(pdf_path: str) -> dict:
    """Extract a structural sample from *pdf_path* and ask the LLM to infer parsing rules."""
    logger.info("Extracting structural sample from %s", pdf_path)
    sample = extract_sample(pdf_path)
    sample_text = "\n".join(
        f"[{i+1}] label={s['label']}, docling_level={s['docling_level']}, text=\"{s['text']}\""
        for i, s in enumerate(sample)
    )
    prompt = CASE_LAW_USER_PROMPT.replace("{sample}", sample_text)
    logger.info("Calling LLM for rule inference (%d elements in sample)", len(sample))
    llm_raw = _call_llm(prompt)
    config_dict = _parse_json(llm_raw)
    logger.info("Domain inferred: %s", config_dict.get("domain", "—"))
    return config_dict


def parse_document(pdf_path: str) -> tuple[dict, list[Node]]:
    """Full pipeline: infer rules then build the document tree. Returns (config, roots)."""
    cfg = infer_config(pdf_path)
    roots = build_tree(pdf_path, cfg)
    return cfg, roots
