"""Generate the actor vocabulary from the acts' own Definitions articles.

The vocabulary is what turns a role from a phrase into an identifier. The extraction stage
emits free text, "The provider or its legal representative" among it, but the classifier has
to enumerate roles in a prompt and Cypher has to match one exactly, and neither works on
prose.

Generated, never written by hand: the subjects come from the definitions the legislature
itself wrote. Committed rather than derived at load time: a regeneration that changes what a
role means then appears as a diff instead of silently moving every role filter.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml
from pydantic import BaseModel, Field

from legal_assistant.graph.queries import NodeQueries
from legal_assistant.obligations.models import Actor
from legal_assistant.rag.prompts import ACTOR_VOCABULARY_PROMPT

logger = logging.getLogger(__name__)

VOCABULARY_PATH = Path(__file__).with_name("actors.yaml")

_NON_WORD = re.compile(r"[^a-z0-9]+")


class DefinedSubject(BaseModel):
    """One obligation-bearing subject, as the model reads it out of a Definitions article."""

    label: str = Field(description="The subject as the regulation names it, singular.")
    defined_in: Optional[str] = Field(
        default=None, description="Id of the paragraph defining it, exactly as given."
    )
    is_a: List[str] = Field(
        default_factory=list,
        description="Labels of broader subjects this one qualifies.",
    )


class DefinedSubjects(BaseModel):
    """Structured output of one generation call, over one act."""

    subjects: List[DefinedSubject] = Field(default_factory=list)


def slugify(label: str) -> str:
    """A stable id for an actor label.

    Ids reach Cypher and a committed file, so they stay lowercase, plain and repeatable.
    """
    return _NON_WORD.sub("_", label.strip().lower()).strip("_")


def _definitions_block(rows: Iterable[dict]) -> str:
    """One definition per line, id first.

    The id is not bracketed or otherwise decorated: whatever punctuation labels an entry, the
    model copies it into ``defined_in`` along with the id.
    """
    return "\n".join(f"{row['id']} {row['text']}" for row in rows)


def _clean_id(value: Optional[str], known_ids: set) -> Optional[str]:
    """An id the model reported, kept only if the query actually returned it.

    Surrounding punctuation is stripped first, because a model echoes the id as it saw it
    labelled. The match itself stays exact, so an invented paragraph is still dropped.
    """
    if not value:
        return None
    candidate = value.strip().strip("[]()<>\"' ")
    return candidate if candidate in known_ids else None


def _as_cross_cutting(actor: Actor) -> Actor:
    """Detach an actor from the act that happened to be read first.

    A subject two acts both define, a Member State or the Commission, bears duties under
    each. Keeping it owned by one would split those duties, and a filter on either half
    would answer incompletely without saying so. Its ``defined_in`` goes too: pointing at
    one act's definition would misrepresent the other's.

    The risk this accepts is a genuine divergence, two acts using one word for different
    roles, which merging would flatten. The vocabulary is committed precisely so that such a
    merge is visible in a diff and can be split by hand.
    """
    return actor.model_copy(update={"celex": None, "defined_in": None})


def generate(graph, llm, acts: List[str]) -> List[Actor]:
    """Read each act's definitions and build the actor vocabulary, sorted by id.

    One call per act: the acts are independent, and keeping them apart is what makes a label
    defined by two of them detectable rather than silently merged.
    """
    structured = llm.with_structured_output(DefinedSubjects)
    by_id: Dict[str, Actor] = {}

    for celex in acts:
        rows = graph.query(NodeQueries.GET_DEFINITIONS_BY_ACTS, params={"acts": [celex]})
        known_ids = {row["id"] for row in rows}

        result = structured.invoke(
            ACTOR_VOCABULARY_PROMPT.format(definitions=_definitions_block(rows))
        )

        for subject in result.subjects:
            actor_id = slugify(subject.label)
            if actor_id in by_id:
                by_id[actor_id] = _as_cross_cutting(by_id[actor_id])
                logger.info(
                    "[vocabulary] %s also defined by %s: treated as cross-cutting",
                    actor_id, celex,
                )
                continue

            by_id[actor_id] = Actor(
                id=actor_id,
                label=subject.label.strip(),
                celex=celex,
                defined_in=_clean_id(subject.defined_in, known_ids),
                is_a=sorted(slugify(parent) for parent in subject.is_a),
            )

        logger.info("[vocabulary] %s: %d subject(s)", celex, len(result.subjects))

    return [by_id[actor_id] for actor_id in sorted(by_id)]


def dumps(actors: Iterable[Actor]) -> str:
    """Serialise deterministically, so a diff always means a real change."""
    payload = [
        {
            "id": actor.id,
            "label": actor.label,
            "celex": actor.celex,
            "defined_in": actor.defined_in,
            "is_a": list(actor.is_a),
        }
        for actor in sorted(actors, key=lambda a: a.id)
    ]
    return yaml.safe_dump(payload, sort_keys=True, allow_unicode=True, default_flow_style=False)


def loads(text: str) -> List[Actor]:
    """Read a serialised vocabulary back."""
    return [Actor(**entry) for entry in yaml.safe_load(text) or []]


def load_vocabulary(path: Path = VOCABULARY_PATH) -> List[Actor]:
    """The committed vocabulary, as the loader and the classifier see it."""
    return loads(path.read_text(encoding="utf-8"))
