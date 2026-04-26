import re
import logging
from dataclasses import dataclass, field

logging.disable(logging.CRITICAL)

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

STRUCTURAL = {"section_header", "title", "chapter", "page_header"}
BODY_LABELS = {"text", "paragraph", "list_item", "caption", "footnote"}

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


def merge_split_fragments(items: list[dict]) -> list[dict]:
    """Repair paragraphs split mid-sentence by PDF two-column reading order."""
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


def extract_sample(pdf_path: str, body_snippet: int = 300) -> list[dict]:
    """Extract structural elements from a PDF for LLM rule inference."""
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

    items = merge_split_fragments(raw)

    for entry in items:
        if entry["label"] not in STRUCTURAL and len(entry["text"]) > body_snippet:
            entry["text"] = entry["text"][:body_snippet] + "…"

    return items


def build_tree(pdf_path: str, config: dict) -> list[Node]:
    """Parse a PDF into a Node tree using LLM-inferred structural rules."""
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = False
    converter = DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=opts)}
    )
    doc = converter.convert(pdf_path).document

    raw: list[dict] = []
    for item, level in doc.iterate_items():
        label = item.label.value if hasattr(item.label, "value") else str(item.label)
        text = item.text.strip().replace("\n", " ") if getattr(item, "text", None) else ""
        if text:
            raw.append({"label": label, "text": text, "docling_level": level})

    items = merge_split_fragments(raw)

    rules = config["rules"]
    roots: list[Node] = []
    stack: list[tuple[int, Node]] = []
    current_node: Node | None = None

    for entry in items:
        label, text = entry["label"], entry["text"]

        if label in STRUCTURAL:
            matched_depth = _match_depth(text, rules)
            rule_matched = matched_depth is not None
            depth = matched_depth if rule_matched else max(0, entry.get("docling_level", 1) - 1)

            node = Node(text=text, label=label, depth=depth, rule_matched=rule_matched)

            while stack and stack[-1][0] >= depth:
                stack.pop()

            if stack:
                stack[-1][1].children.append(node)
            else:
                roots.append(node)

            stack.append((depth, node))
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
