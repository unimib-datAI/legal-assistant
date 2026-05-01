import json
import logging
import re

import config
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from service.case_law.doc_parser import extract_sample, build_tree, Node
from service.rag.prompt import (CASE_LAW_DOCUMENT_PARSING_SYSTEM_PROMPT, CASE_LAW_DOCUMENT_PARSING_USER_PROMPT,
                                CASE_LAW_ENTITY_SUMMARY_SYSTEM_PROMPT, CASE_LAW_ENTITY_SUMMARY_USER_PROMPT,
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

def parse_document(pdf_path: str) -> tuple[dict, list[Node]]:
    """Full pipeline: infer rules then build the document tree. Returns (parsing_rules, roots)."""
    parsing_rules = _infer_config(pdf_path)
    roots = build_tree(pdf_path, parsing_rules)
    return parsing_rules, roots

def summarize_document(pdf_path: str, char_length: int = 150) -> str:
    """Produce a high-level summary of the entire document from the raw PDF text."""
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = False
    converter = DocumentConverter(format_options={"pdf": PdfFormatOption(pipeline_options=opts)})
    doc = converter.convert(pdf_path).document
    document_content = "\n".join(
        item.text.strip().replace("\n", " ")
        for item, _ in doc.iterate_items()
        if getattr(item, "text", None)
    )
    user_prompt = CASE_LAW_ENTIRE_DOC_SUMMARY_USER_PROMPT.format(
        char_length=char_length,
        document_content=document_content,
    )
    logger.info("Summarising full document from raw PDF: %s", pdf_path)
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
    if graph.node_exists("CaseLaw", case_law_id):
        logger.info("CaseLaw node %s already exists — reusing it", celex)
    else:
        graph.create_graph_node("CaseLaw", {"id": case_law_id, "celex": celex})
        logger.info("Created CaseLaw node: %s", celex)

    topics_section = next((s for s in flat if s["heading"] == "Topics"), None)
    if topics_section:
        for topic in topics_section.get("body", []):
            topic = topic.strip()
            if not topic:
                continue
            topic_id = f"case_law_topic:{celex}:{topic}"
            graph.create_graph_node("CaseLawTopic", {"id": topic_id, "label": topic, "celex": celex})
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
        graph.create_graph_node("CaseLawSection", {
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
