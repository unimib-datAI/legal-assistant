"""Pluggable RAG-method abstraction.

A :class:`RagMethod` wraps one retrieval strategy behind a uniform interface so
the frontend can switch between methods at runtime and auto-generate their
hyperparameter controls from :meth:`RagMethod.param_specs`.

The attribution dataclasses (:class:`SourceRef`, :class:`Segment`,
:class:`AttributedAnswer`) describe a synthesised answer broken into atomic
claims, each linked to the exact retrieved passages that support it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from langchain_core.retrievers import BaseRetriever


@dataclass(frozen=True)
class ParamSpec:
    """Declarative description of one tunable hyperparameter.

    ``name`` must match the corresponding pydantic field on the retriever, so a
    config dict built from these specs can be splatted into the constructor.
    The frontend renders a widget per ``kind``: ``bool`` -> toggle,
    ``int``/``float`` -> slider.
    """

    name: str
    label: str
    kind: Literal["bool", "int", "float"]
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    help: str = ""


@dataclass(frozen=True)
class SourceRef:
    """A single retrieved passage, numbered with a stable marker (e.g. ``S1``)."""

    marker: str
    doc_id: str
    act: str
    title: str
    type: str
    text: str


@dataclass
class Segment:
    """One atomic claim of the answer and the source markers that support it."""

    text: str
    source_markers: List[str] = field(default_factory=list)


@dataclass
class AttributedAnswer:
    """A synthesised answer with per-claim source attribution."""

    segments: List[Segment]
    sources: List[SourceRef]
    raw_answer: str


class RagMethod(ABC):
    """A named retrieval strategy that builds a configured retriever on demand.

    Subclasses declare their tunable hyperparameters via :meth:`param_specs` and
    turn a config dict (keyed by ``ParamSpec.name``) into a ready retriever via
    :meth:`build_retriever`. Shared, expensive resources (graph, vector stores,
    classifier, LLMs) come from the injected ``ctx`` rather than being rebuilt.
    """

    id: str
    name: str
    description: str = ""

    @abstractmethod
    def param_specs(self) -> List[ParamSpec]:
        """Return the tunable hyperparameters this method exposes."""

    @abstractmethod
    def build_retriever(self, ctx: "Any", config: Dict[str, Any]) -> BaseRetriever:
        """Instantiate the underlying retriever from shared resources + config."""

    def default_config(self) -> Dict[str, Any]:
        """Config dict of every spec's default — the UI's starting state."""
        return {spec.name: spec.default for spec in self.param_specs()}
