"""Pluggable RAG-method abstraction — the contract for a retrieval strategy.

A :class:`RagMethod` wraps one retrieval strategy behind a uniform interface so
callers can switch between methods at runtime and auto-generate their
hyperparameter controls from :meth:`RagMethod.param_specs`.

**To add a strategy:** write the retriever under ``rag/retrievers/``, subclass
:class:`RagMethod` in ``rag/methods/<name>.py``, and add an instance to ``_METHODS``
in :mod:`legal_assistant.rag.methods.registry`. Nothing else needs to change — the
CLI, the eval scripts, and the frontend all read the registry.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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
