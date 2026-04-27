import re
import logging
from dataclasses import dataclass, field
from typing import NamedTuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.disable(logging.CRITICAL)

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

STRUCTURAL = {"section_header", "title", "chapter", "page_header"}
BODY_LABELS = {"text", "paragraph", "list_item", "caption", "footnote"}

_DASH = r"[-–]"
_TOPICS_PATTERN = re.compile(r"^\(.+" + _DASH + r".+\)$", re.DOTALL)
_TOPIC_SPLIT = re.compile(r"\s+" + _DASH + r"\s*(?=[A-Z’’’])")
_GIVES_FOLLOWING = re.compile(r"gives the following", re.IGNORECASE)
_JUDGMENT_HEADER = re.compile(r"^judgment$", re.IGNORECASE)

_DANGLING_WORDS = frozenset({
    "a", "an", "the", "of", "in", "to", "for", "with", "by", "on", "at",
    "from", "as", "into", "through", "before", "after", "between", "under",
    "over", "upon", "and", "or", "but", "nor", "its", "their", "this",
    "that", "which", "who", "where", "whether", "both", "either", "such",
    "any", "all", "each", "other", "another", "within", "without", "against",
    "among", "per", "than", "about", "around", "since", "until", "unless",
})
_TERMINAL_PUNCT = frozenset(".!?\"'»'")


def _is_fragment(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped[-1] in _TERMINAL_PUNCT or stripped == "…":
        return False
    last_word = stripped.rsplit(None, 1)[-1].lower().rstrip(",;:")
    return last_word in _DANGLING_WORDS


def _merge_split_fragments(items: list[dict]) -> list[dict]:
    """Repair paragraphs split mid-sentence due to PDF format:
    items = [
    {"label": "text", "text": "The supervisory authority shall not be competent to supervise processing operations of courts acting in their judicial", "docling_level": 2},
    {"label": "text", "text": "ECLI:EU:C:2022:216",  "docling_level": 1},   # page footer (column 2)
    {"label": "text", "text": "capacity.",            "docling_level": 2},
]
    """
    result = list(items)
    i = 0
    while i < len(result):
        if _is_fragment(result[i].get("text", "")):
            for j in range(i + 1, min(i + 25, len(result))):
                ctext = result[j].get("text", "").strip()
                if not ctext:
                    continue
                if ctext[0].islower():
                    result[i] = {**result[i], "text": result[i]["text"].strip() + " " + ctext}
                    result.pop(j)
                    break
        i += 1
    return result


@dataclass
class Node:
    text: str
    label: str
    depth: int
    children: list = field(default_factory=list)
    body: list = field(default_factory=list)
    rule_matched: bool = True

    def all_text(self) -> str:
        parts = [self.text] + [c.all_text() for c in self.children]
        return "\n".join(filter(None, parts))


def extract_sample(pdf_path: str, body_snippet: int = 300) -> list[dict]:
    """Extract structural elements from a PDF for LLM rule inference.
    Example:
    [1] label=section_header, docling_level=1, text="Reports of Cases"
    [2] label=section_header, docling_level=1, text="JUDGMENT OF THE COURT (First Chamber)"
    [3] label=text, docling_level=2, text="24 March 2022*"
    """
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = False
    converter = DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=opts)}
    )
    doc = converter.convert(pdf_path).document

    raw = []
    for item, level in doc.iterate_items():
        label = item.label.value if hasattr(item.label, "value") else str(item.label)
        text = item.text.strip().replace("\n", " ") if getattr(item, "text", None) else ""
        if label in STRUCTURAL:
            raw.append({"label": label, "docling_level": level, "text": text})
        elif label in BODY_LABELS and text:
            snippet = text[:body_snippet] + ("…" if len(text) > body_snippet else "")
            raw.append({"label": label, "docling_level": level, "text": snippet})

    items = _merge_split_fragments(raw)

    for entry in items:
        if entry["label"] not in STRUCTURAL and len(entry["text"]) > body_snippet:
            entry["text"] = entry["text"][:body_snippet] + "…"

    return items


def build_tree(pdf_path: str, parsing_rules: dict) -> list[Node]:
    """Parse a PDF into a Node tree using LLM-inferred structural rules:
    1. Convert the PDF into a flat list of text elements -> item.label = DocItemLabel.SECTION_HEADER  item.text = "Reports of Cases"
    2. For each element merge the paragraphs that are likely split mid-sentence by the PDF formatting (e.g. two-column layout).
    3. Detect and reconstruct the preamble sections in EU case law (Topics + General Information).
    4. Iterate through the cleaned flat list and build a tree based on the LLM-inferred rules: if an element matches a
    structural rule, create a new Node with depth based on the rule or docling level
    """
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = False
    converter = DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=opts)}
    )
    case_law = converter.convert(pdf_path).document

    raw_docling_items: list[dict] = []
    for docling_item, level in case_law.iterate_items():
        label = docling_item.label.value if hasattr(docling_item.label, "value") else str(docling_item.label)
        text = docling_item.text.strip().replace("\n", " ") if getattr(docling_item, "text", None) else ""

        if text:
            raw_docling_items.append({"label": label, "text": text, "docling_level": level})

    final_items = _merge_split_fragments(raw_docling_items)
    final_items = _extract_preamble_sections(final_items)

    class _StackEntry(NamedTuple):
        depth: int
        node: Node

    rules = parsing_rules["rules"]
    roots: list[Node] = []
    stack: list[_StackEntry] = []
    current_node: Node | None = None

    for entry in final_items:
        label, text = entry["label"], entry["text"]

        if label in STRUCTURAL:
            matched_depth = _match_depth(text, rules)
            rule_matched = matched_depth is not None
            depth = matched_depth if rule_matched else max(0, entry.get("docling_level", 1) - 1)
            node = Node(text=text, label=label, depth=depth, rule_matched=rule_matched)

            while stack and stack[-1].depth >= depth:
                stack.pop()

            if stack:
                stack[-1].node.children.append(node)
            else:
                roots.append(node)

            stack.append(_StackEntry(depth, node))
            current_node = node

        elif label in BODY_LABELS and current_node is not None:
            current_node.body.append(text)

    return roots


def flatten(roots: list[Node]) -> list[dict]:
    out: list[dict] = []

    def _walk(nodes: list[Node]) -> None:
        for node in nodes:
            out.append({"heading": node.text, "depth": node.depth, "body": node.body})
            _walk(node.children)

    _walk(roots)
    return out

def _match_depth(text: str, rules: list[dict]) -> int | None:
    for rule in rules:
        pattern, rtype = rule["pattern"], rule.get("type", "prefix")
        if rtype == "regex" or pattern.startswith("^") or pattern.endswith("$"):
            try:
                if re.match(pattern, text.strip(), re.IGNORECASE):
                    return rule["depth"]
                continue
            except re.error:
                pass
        if rtype == "prefix" and text.lower().startswith(pattern.lower()):
            return rule["depth"]
        if rtype == "exact" and text.strip().lower() == pattern.lower():
            return rule["depth"]
    return None


def _find_gen_info_end(items: list[dict], after: int) -> tuple[int, bool]:
    """
    Return (end_index, inclusive) marking the boundary of the preamble after the topics block.

    Priority:
      1. Element containing "gives the following" — include it in General Information (inclusive).
      2. Structural header with text "Judgment" — exclude it (exclusive).
      3. First structural header found after `after` — exclude it (exclusive).
    Returns (len(items), False) if nothing is found.
    """
    gives_idx = next(
        (i for i in range(after, len(items)) if _GIVES_FOLLOWING.search(items[i].get("text", ""))),
        None,
    )
    if gives_idx is not None:
        return gives_idx, True

    judgment_idx = next(
        (i for i in range(after, len(items))
         if items[i].get("label") in STRUCTURAL
         and _JUDGMENT_HEADER.match(items[i].get("text", "").strip())),
        None,
    )
    if judgment_idx is not None:
        return judgment_idx, False

    first_struct_idx = next(
        (i for i in range(after, len(items)) if items[i].get("label") in STRUCTURAL),
        None,
    )
    if first_struct_idx is not None:
        return first_struct_idx, False

    return len(items), False


def _make_section_header(text: str) -> dict:
    return {"label": "section_header", "text": text, "docling_level": 1}


def _make_body_item(text: str) -> dict:
    return {"label": "text", "text": text, "docling_level": 2}


def _demote_to_text(item: dict) -> dict:
    return {**item, "label": "text"}


def _expand_topics_block(raw_text: str) -> list[dict]:
    """Turn the parenthetical topic string into a Topics header + one body item per topic."""
    cleaned = raw_text.strip().lstrip("(").rstrip(")")
    topic_strings = [t.strip() for t in _TOPIC_SPLIT.split(cleaned) if t.strip()]
    return [_make_section_header("Topics")] + [_make_body_item(t) for t in topic_strings]


def _extract_preamble_sections(items: list[dict]) -> list[dict]:
    """
    Restructure the EU case law preamble into three synthetic sections:

      - The first structural heading (e.g. "Reports of Cases") keeps its role;
        everything between it and the topics block is demoted to body text.
      - The parenthetical topics block becomes a "Topics" section.
      - Everything from after Topics until "gives the following" (inclusive)
        becomes a "General Information" section; bold party names inside it
        are demoted to body text.
    """
    topics_idx = next(
        (i for i, it in enumerate(items) if _TOPICS_PATTERN.match(it.get("text", "").strip())),
        None,
    )
    if topics_idx is None:
        return items

    document_opener_idx = next(
        (i for i in range(topics_idx) if items[i].get("label") in STRUCTURAL),
        None,
    )
    preamble_end_idx, preamble_end_inclusive = _find_gen_info_end(items, topics_idx + 1)

    result = list(items)

    # Step 1 — demote headings between the document opener and the topics block
    # (e.g. "JUDGMENT OF THE COURT (Grand Chamber)" → body of "Reports of Cases")
    if document_opener_idx is not None:
        for i in range(document_opener_idx + 1, topics_idx):
            if result[i].get("label") in STRUCTURAL:
                result[i] = _demote_to_text(result[i])

    # Step 2 — replace the topics block with a Topics section header + individual items
    topics_items = _expand_topics_block(result[topics_idx]["text"])
    result = result[:topics_idx] + topics_items + result[topics_idx + 1:]

    # Adjust preamble_end_idx to account for the items inserted in step 2
    items_inserted = len(topics_items) - 1
    preamble_end_idx += items_inserted

    # Step 3 — inject a General Information section covering the rest of the preamble
    gen_info_start = topics_idx + len(topics_items)
    gen_info_end = preamble_end_idx + 1 if preamble_end_inclusive else preamble_end_idx

    if gen_info_start < gen_info_end:
        for i in range(gen_info_start, gen_info_end):
            if result[i].get("label") in STRUCTURAL:
                result[i] = _demote_to_text(result[i])
        result.insert(gen_info_start, _make_section_header("General Information"))

    return result
