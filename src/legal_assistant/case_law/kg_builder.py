"""Writes a parsed CJEU judgment into Neo4j.

The retrieval unit is the **numbered judgment paragraph**, not the section. A section can
run to 46 paragraphs (tens of KB)   far too coarse to embed or to put in a prompt   and the
paragraph is anyway the unit the Court itself is cited by ("see paragraph 71 of that
judgment"), which is also how ``case_law_golden_dataset.csv`` records its sources.

Graph structure::

    (CaseLaw {celex, case_number})
      -[:HAS_TOPIC]->   (CaseLawTopic {label})
      -[:HAS_SECTION]-> (CaseLawSection {heading, depth, path})
                          -[:CONTAINS]->      (CaseLawSection)
                          -[:HAS_PARAGRAPH]-> (CaseLawParagraph {number, text, is_operative})

The text property is named ``text``  not ``body``  so ``Neo4jGraph.generate_text_embeddings``
and ``Neo4jVector.from_existing_graph``, both of which read ``n.text``, work on it unchanged.
"""
import logging
import re
from dataclasses import dataclass

from legal_assistant.case_law.tree import Node, flatten

logger = logging.getLogger(__name__)

# Sections that carry no judicial reasoning: EUR-Lex boilerplate (ECLI, court composition,
# language of the case) and the topic index, which becomes CaseLawTopic edges instead.
PREAMBLE_SECTIONS = frozenset({"Reports of Cases", "Topics", "General Information"})

# "71 That said, it must be emphasised that..."  _linearize merges the number cell and the
# prose cell of the source table, so the number arrives at the head of the string.
_PARAGRAPH_NUM = re.compile(r"^(\d+)\s+")

# The operative part is not a section of its own: it sits at the tail of "Costs", introduced
# by "On those grounds, the Court (Grand Chamber) hereby rules:" and followed by the rulings.
_OPERATIVE_ANCHOR = re.compile(r"hereby (?:rules|orders|declares)", re.IGNORECASE)
# "1. Article 58(5) of Regulation 2016/679 must be interpreted as meaning that..."
_OPERATIVE_NUM = re.compile(r"^(\d+)[.)]\s+")
_SIGNATURES = re.compile(r"^\[?\s*signatures?\s*\]?\.?$", re.IGNORECASE)

# 62019CJ0645 -> C-645/19.  Sector 6, 4-digit year, CJ/CO (Court of Justice) or TJ/TO
# (General Court), then the zero-padded case number.
_CELEX_CASE = re.compile(r"^6(\d{4})(C|T)[A-Z](\d{4})$", re.IGNORECASE)


def celex_to_case_number(celex: str) -> str:
    """``62019CJ0645`` -> ``C-645/19``. Falls back to the CELEX when it doesn't parse."""
    match = _CELEX_CASE.match(celex.strip())
    if not match:
        return celex
    year, court, number = match.groups()
    return f"{court.upper()}-{int(number)}/{year[2:]}"


@dataclass(frozen=True)
class CaseLawParagraph:
    """One numbered judgment paragraph, or one item of the operative part."""
    id: str
    number: int | None
    text: str
    is_operative: bool


def split_paragraphs(celex: str, body: list[str]) -> list[CaseLawParagraph]:
    """Split a section's body items into paragraph nodes.

    Three item kinds are interleaved in ``body``:

    * **numbered paragraphs**  the judgment's own reasoning, the citable unit.
    * **unnumbered continuations**  block quotes of the legislation under discussion,
      following the paragraph that introduces them ("4 Recital 10 of the GDPR states:").
      They are merged into that paragraph: floating free they would put verbatim regulation
      text into the case law index, where it would read as if the Court had said it.
    * **the operative part**  everything after the "hereby rules" anchor.

    Unnumbered items with no preceding paragraph are dropped (headnote fragments).
    """
    paragraphs: list[CaseLawParagraph] = []
    buffer: list[str] = []
    number: int | None = None
    operative = False
    operative_seq = 0

    def flush() -> None:
        nonlocal buffer, number
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        if text:
            if operative:
                node_id = f"{celex}_op_{number}"
            else:
                node_id = f"{celex}_par_{number}"
            paragraphs.append(
                CaseLawParagraph(id=node_id, number=number, text=text, is_operative=operative)
            )
        buffer, number = [], None

    for item in body:
        if not operative and _OPERATIVE_ANCHOR.search(item):
            flush()
            operative = True
            continue

        if operative:
            if _SIGNATURES.match(item.strip()):
                continue
            match = _OPERATIVE_NUM.match(item)
            if match:
                flush()
                operative_seq += 1
                number = int(match.group(1))
                buffer = [item[match.end():].strip()]
            elif buffer:
                buffer.append(item)
            else:
                # A single-ruling judgment states its holding without a "1." prefix.
                operative_seq += 1
                number = operative_seq
                buffer = [item]
            continue

        match = _PARAGRAPH_NUM.match(item)
        if match:
            flush()
            number = int(match.group(1))
            buffer = [item[match.end():].strip()]
        elif buffer:
            buffer.append(item)

    flush()
    return paragraphs


def _section_paths(flat: list[dict]) -> list[str]:
    """Positional path per section ("5", "5.1", …).

    Used as the section id, so the id is stable under a re-parse that changes a heading
    unlike the old scheme, which baked the flat index and a 60-char heading prefix into it.
    """
    paths: list[str] = []
    stack: list[int] = []

    for section in flat:
        depth = section["depth"]
        if depth < len(stack):
            stack = stack[: depth + 1]
            stack[depth] += 1
        else:
            while len(stack) < depth:  # a skipped level still gets a slot
                stack.append(0)
            stack.append(0)
        paths.append(".".join(str(i) for i in stack))

    return paths


def create_case_law_kg(
    celex: str,
    flat: list[dict],
    graph,
    summaries: list[dict] | None = None,
    *,
    strict: bool = True,
) -> dict:
    """Validate and write the case law subgraph for one judgment. Returns counts.

    The judgment is built in memory first and checked against the parsed tree: every body
    item of a substantive section must end up inside a ``CaseLawParagraph``. Only then is
    anything written to ``graph``. With ``strict=False`` violations are logged and the write
    proceeds anyway.

    This is the gate for **both** write paths  the ingest pipeline (through
    :func:`build_from_tree`) and the Streamlit parser page call it.
    """
    # Imported here: the validation package imports kg_builder for its regexes, so a
    # module-level import would be circular.
    from legal_assistant.validation import case_law_source
    from legal_assistant.validation.gate import build_validated

    plan = build_validated(
        lambda recorder: _write_case_law_kg(celex, flat, recorder, summaries),
        root_id=celex,
        label=f"case law {celex}",
        source_inventory=case_law_source.body_fragments(flat),
        reconstructed=case_law_source.paragraph_texts,
        exempt=case_law_source.body_exemptions(flat),
        conservation_kind="body",
        strict=strict,
    )
    plan.replay(graph)
    return _counts_from_plan(plan, celex)


def _counts_from_plan(plan, celex: str) -> dict:
    """Per-judgment counts, read off the validated plan."""
    paragraphs = [n for n in plan.node_ops if n.label == "CaseLawParagraph"]
    counts = {
        "sections": sum(1 for n in plan.node_ops if n.label == "CaseLawSection"),
        "paragraphs": len(paragraphs),
        "operative": sum(1 for n in paragraphs if n.properties.get("is_operative")),
    }
    logger.info(
        "[KG] %s (%s): %d sections, %d paragraphs (%d operative)",
        celex, celex_to_case_number(celex),
        counts["sections"], counts["paragraphs"], counts["operative"],
    )
    return counts


def _write_case_law_kg(
    celex: str,
    flat: list[dict],
    graph,
    summaries: list[dict] | None = None,
) -> dict:
    """Emit the case law subgraph into ``graph``. No validation  see the caller.

    ``summaries`` is optional: paragraph-level retrieval does not need it, and generating it
    costs one LLM call per section. When given, it is the list produced by
    ``llm_orchestrator.summarize_section`` plus a leading "Document Summary" entry.
    """
    by_heading: dict[str, str] = {s["heading"]: s.get("summary", "") for s in (summaries or [])}
    doc_summary = by_heading.pop("Document Summary", "")

    graph.upsert_graph_node("CaseLaw", {
        "id": celex,
        "celex": celex,
        "case_number": celex_to_case_number(celex),
        "summary": doc_summary,
    })

    _write_topics(celex, flat, graph)
    return _write_sections(celex, flat, graph, by_heading)


def _write_topics(celex: str, flat: list[dict], graph) -> None:
    section = next((s for s in flat if s["heading"] == "Topics"), None)
    if not section:
        return

    for topic in section.get("body", []):
        topic = topic.strip()
        if not topic:
            continue
        topic_id = f"case_law_topic:{celex}:{topic}"
        graph.upsert_graph_node("CaseLawTopic", {"id": topic_id, "label": topic, "celex": celex})
        graph.create_relationship("CaseLaw", "CaseLawTopic", celex, topic_id, "HAS_TOPIC")


def _write_sections(celex: str, flat: list[dict], graph, by_heading: dict[str, str]) -> dict:
    paths = _section_paths(flat)
    # depth -> id of the most recent section at that depth, for parent resolution.
    open_at_depth: dict[int, str] = {}
    counts = {"sections": 0, "paragraphs": 0, "operative": 0}

    for section, path in zip(flat, paths):
        heading, depth = section["heading"], section["depth"]
        if heading == "Topics":
            continue

        section_id = f"{celex}_sec_{path}"
        graph.upsert_graph_node("CaseLawSection", {
            "id": section_id,
            "celex": celex,
            "heading": heading,
            "depth": depth,
            "path": path,
            "summary": by_heading.get(heading, ""),
        })
        counts["sections"] += 1

        parent_id = next(
            (open_at_depth[d] for d in range(depth - 1, -1, -1) if d in open_at_depth), None
        )
        if parent_id is None:
            graph.create_relationship("CaseLaw", "CaseLawSection", celex, section_id, "HAS_SECTION")
        else:
            graph.create_relationship(
                "CaseLawSection", "CaseLawSection", parent_id, section_id, "CONTAINS"
            )

        open_at_depth[depth] = section_id
        for deeper in [d for d in open_at_depth if d > depth]:
            del open_at_depth[deeper]

        if heading in PREAMBLE_SECTIONS:
            continue

        for paragraph in split_paragraphs(celex, section.get("body", [])):
            graph.upsert_graph_node("CaseLawParagraph", {
                "id": paragraph.id,
                "celex": celex,
                "number": paragraph.number,
                "text": paragraph.text,
                "section_heading": heading,
                "is_operative": paragraph.is_operative,
            })
            graph.create_relationship(
                "CaseLawSection", "CaseLawParagraph", section_id, paragraph.id, "HAS_PARAGRAPH"
            )
            counts["paragraphs"] += 1
            counts["operative"] += int(paragraph.is_operative)

    return counts


def build_from_tree(
    celex: str,
    roots: list[Node],
    graph,
    summaries: list[dict] | None = None,
    *,
    strict: bool = True,
) -> dict:
    """Convenience wrapper: flatten a parsed tree, validate it, and write it."""
    return create_case_law_kg(
        celex=celex, flat=flatten(roots), graph=graph, summaries=summaries, strict=strict
    )
