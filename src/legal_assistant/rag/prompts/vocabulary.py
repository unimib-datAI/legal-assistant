"""Prompt for generating the actor vocabulary from an act's Definitions article."""
from datetime import date

from legal_assistant.rag.prompts.registry import PromptVersion, registry


ACTOR_VOCABULARY_V1 = """You are an expert in EU legislation.

Below are the numbered definitions of one EU regulation, each preceded by the id of the
paragraph it appears in. Report every defined term that can BEAR A LEGAL OBLIGATION.

A term qualifies only if it denotes someone capable of acting and of being held to a duty: a
natural or legal person, a body, an authority, an institution, or a Member State. Apply that
test to the definitions in front of you. Do not report a term because a similar regulation
defines one, and do not report a term this regulation does not define.

A term does not qualify if it denotes a thing, an activity, a document, a category of data, a
state of affairs or a property, however central it is to the regulation. If the definition
says what something IS rather than who someone IS, leave it out.

For each qualifying term, report:

- label: the term exactly as this regulation writes it, singular.
- defined_in: the id of the paragraph defining it, copied from the list below. Never invent
  an id, and never cite a paragraph that is not listed.
- is_a: broader terms it falls under, drawn only from labels you also report. Report one only
  where THIS regulation's own wording establishes it, which typically means a definition
  written as a list of others: "'X' means an A, a B or a C" makes A, B and C each an X. Leave
  it empty otherwise. Do not infer a hierarchy from what the words ordinarily mean.

=== DEFINITIONS ===
{definitions}
"""

ACTOR_VOCABULARY_V2 = """You are an expert in EU legislation.

Below are the numbered definitions of one EU regulation, each preceded by the id of the
paragraph it appears in. Report every defined term that can be a PARTY TO AN OBLIGATION,
whether it BEARS the obligation or BENEFITS from it.

A term qualifies only if it denotes a party that can hold or be owed a duty: a natural or
legal person, a body, an authority, an institution, a Member State, or the individual a duty
protects. Apply that test to the definitions in front of you. Do not report a term because a
similar regulation defines one, and do not report a term this regulation does not define.

Include the protected individual even though such a party bears no duties of its own: it is
the beneficiary of most of the regulation's obligations, and the analysis needs to name it.

A term does not qualify if it denotes a thing, an activity, a document, a category of data, a
state of affairs or a property, however central it is to the regulation. If the definition
says what something IS rather than who someone IS, leave it out.

For each qualifying term, report:

- label: the term exactly as this regulation writes it, singular.
- defined_in: the id of the paragraph defining it, copied from the list below. Never invent
  an id, and never cite a paragraph that is not listed.
- is_a: broader terms it falls under, drawn only from labels you also report. Report one only
  where THIS regulation's own wording establishes it. That happens in two shapes: a
  definition written as a list of others, "'X' means an A, a B or a C", makes A, B and C each
  an X; and a definition written as a qualified other, "'X' means a Y which ...", makes X a Y.
  Leave it empty otherwise. Do not infer a hierarchy from what the words ordinarily mean.

=== DEFINITIONS ===
{definitions}
"""

registry.register(PromptVersion(
    name="actor_vocabulary",
    version="v1",
    created=date(2026, 7, 23),
    notes="Initial version. States the obligation-bearing test as a criterion rather than "
          "enumerating expected roles, which biased the model toward reporting another "
          "act's vocabulary.",
    body=ACTOR_VOCABULARY_V1,
    active=False,
))

registry.register(PromptVersion(
    name="actor_vocabulary",
    version="v2",
    created=date(2026, 7, 23),
    notes="Widen the criterion from obligation-BEARING to party-to-an-obligation, so a "
          "beneficiary-only subject such as the GDPR data subject is kept: Actor nodes serve "
          "BENEFITS as well as ADDRESSED_TO, and the data subject is the commonest "
          "beneficiary. Also name the second IS_A shape explicitly, \"means a Y which ...\", "
          "which gpt-5-mini caught on 'supervisory authority concerned' and v1 did not spell "
          "out.",
    body=ACTOR_VOCABULARY_V2,
    active=True,
))
