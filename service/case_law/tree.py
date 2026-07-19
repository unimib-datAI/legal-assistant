"""Document tree primitives shared by the case law parsers.

Kept dependency-free (no docling, no network) so both the HTML parser and the
legacy PDF parser can build the same ``Node`` contract that ``flatten()`` and
``kg_builder.create_case_law_kg()`` consume.
"""
from dataclasses import dataclass, field
from typing import NamedTuple


@dataclass
class Node:
    text: str
    label: str
    depth: int
    children: list["Node"] = field(default_factory=list)
    body: list[str] = field(default_factory=list)
    rule_matched: bool = True

    def all_text(self) -> str:
        parts = [self.text] + [c.all_text() for c in self.children]
        return "\n".join(filter(None, parts))


class _StackEntry(NamedTuple):
    depth: int
    node: Node


class TreeBuilder:
    """Builds a Node tree from a linear (heading, depth) / body stream.

    Encapsulates the stack discipline: a new node at depth *d* pops every open
    ancestor at depth >= d, then attaches to whatever remains open. A node whose
    depth skips a level (e.g. 0 -> 2) still attaches to the nearest shallower
    ancestor rather than failing.
    """

    def __init__(self) -> None:
        self.roots: list[Node] = []
        self._stack: list[_StackEntry] = []
        self._current: Node | None = None

    def open_section(self, text: str, depth: int, label: str = "section_header") -> Node:
        node = Node(text=text, label=label, depth=depth)

        while self._stack and self._stack[-1].depth >= depth:
            self._stack.pop()

        if self._stack:
            self._stack[-1].node.children.append(node)
        else:
            self.roots.append(node)

        self._stack.append(_StackEntry(depth, node))
        self._current = node
        return node

    def add_body(self, text: str) -> None:
        if self._current is not None:
            self._current.body.append(text)


def flatten(roots: list[Node]) -> list[dict]:
    """Pre-order flattening. `depth` is what the KG builder re-parents on."""
    out: list[dict] = []

    def _walk(nodes: list[Node]) -> None:
        for node in nodes:
            out.append({"heading": node.text, "depth": node.depth, "body": node.body})
            _walk(node.children)

    _walk(roots)
    return out
