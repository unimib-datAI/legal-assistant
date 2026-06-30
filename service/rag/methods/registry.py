"""Registry of available RAG methods.

To add a new method: implement a :class:`~service.rag.methods.base.RagMethod`
subclass and append an instance to ``_METHODS`` below — the frontend picks it up
automatically.
"""
from __future__ import annotations

from typing import Dict, List

from service.rag.methods.base import RagMethod
from service.rag.methods.hybrid_method import HybridRagMethod
from service.rag.methods.topics_method import TopicsRagMethod

_METHODS: List[RagMethod] = [
    HybridRagMethod(),
    TopicsRagMethod(),
]

REGISTRY: Dict[str, RagMethod] = {m.id: m for m in _METHODS}


def list_methods() -> List[RagMethod]:
    """Return all registered methods in registration order."""
    return list(_METHODS)


def get_method(method_id: str) -> RagMethod:
    """Return the method for ``method_id`` or raise ``KeyError``."""
    return REGISTRY[method_id]
