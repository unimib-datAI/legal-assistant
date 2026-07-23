"""Registry of available RAG methods.

To add a new method: implement a :class:`~legal_assistant.rag.methods.base.RagMethod`
subclass and append an instance to ``_METHODS`` below; the frontend picks it up
automatically.
"""
from __future__ import annotations

from typing import Dict, List

from legal_assistant.rag.methods.base import RagMethod
from legal_assistant.rag.methods.hybrid import HybridRagMethod
from legal_assistant.rag.methods.topics import TopicsRagMethod

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
