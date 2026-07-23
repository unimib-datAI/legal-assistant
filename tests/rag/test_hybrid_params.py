"""Guard on the tunable surface: a ParamSpec must name a field the retriever actually has."""
from __future__ import annotations

from legal_assistant.rag.methods.hybrid import _HYDE_ITERATIONS, HybridRagMethod
from legal_assistant.rag.retrievers.hybrid import HybridRetriever


def test_every_param_matches_a_retriever_field():
    """A ParamSpec naming a field that does not exist is silently ignored at construction."""
    fields = set(HybridRetriever.model_fields)
    names = {spec.name for spec in HybridRagMethod().param_specs()}
    unknown = sorted(name for name in names - fields if name != _HYDE_ITERATIONS)
    assert unknown == []


def test_annexes_are_not_a_tunable_parameter():
    """The annex branch is a capability, not a knob to A/B: it stays out of the eval sweep."""
    names = {spec.name for spec in HybridRagMethod().param_specs()}
    assert not {"use_annexes", "top_k_annexes", "annex_score_threshold"} & names
