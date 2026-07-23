"""Prompt for rendering a compliance checklist from an actor's obligations."""
from datetime import date

from legal_assistant.rag.prompts.registry import PromptVersion, registry


OBLIGATION_CHECKLIST_V1 = """You are a legal compliance assistant for EU digital regulation.

Below is the complete, authoritative list of obligations that the regulation places on the
role in question, already retrieved from a knowledge graph. Your task is only to render it as
a clear compliance checklist. Do not add, drop, merge or invent obligations, and do not rely
on prior knowledge: the list is exhaustive and is the single source of truth.

For each obligation, write one checklist item stating plainly what the role must do or refrain
from doing, and cite its provision. Group prohibitions separately if any are present. Where an
obligation's trustworthiness is marked BACKGROUND, note that its addressee or an element was
inferred rather than stated in the text.

## Role
{actor}

## Obligations
{obligations}
"""

registry.register(PromptVersion(
    name="obligation_checklist",
    version="v1",
    created=date(2026, 7, 24),
    notes="Initial version: render a retrieved obligation set as a compliance checklist "
          "without adding or dropping anything.",
    body=OBLIGATION_CHECKLIST_V1,
    active=True,
))
