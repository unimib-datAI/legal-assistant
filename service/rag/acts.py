"""Single source of truth for the four EU acts: CELEX ids, display names, and the
surface keyword lexicon used to recognise an act by name or regulation number.

Consumers:
- ``rag_alternative.py`` / ``engine.py`` — CELEX -> display name for passage headers.
- ``intent_classifier.py`` — ``acts_mentioned_in`` as a deterministic floor: an act the
  user names explicitly is force-included regardless of the LLM's relevance score.
- ``test/rag_eval/retrieval_eval.py`` — ``act_to_celex`` maps a dataset act label to CELEX.
"""
from typing import List, Optional

CELEX_TO_ACT_NAME = {
    "32022R0868": "Data Governance Act",
    "32023R2854": "Data Act",
    "32024R1689": "AI Act",
    "32016R0679": "GDPR",
}

# Surface keyword -> CELEX. Matched as a lowercase substring against a query or a
# dataset act label. Order matters: the more specific "data governance act" and the
# "governance" cue precede "data act" so a Data Governance Act label/mention is never
# misread as the Data Act.
ACT_NAME_KEYWORDS = (
    ("data governance act", "32022R0868"),
    ("governance", "32022R0868"),
    ("2022/868", "32022R0868"),
    ("general data protection", "32016R0679"),
    ("gdpr", "32016R0679"),
    ("2016/679", "32016R0679"),
    ("data act", "32023R2854"),
    ("2023/2854", "32023R2854"),
    ("artificial intelligence act", "32024R1689"),
    ("ai act", "32024R1689"),
    ("2024/1689", "32024R1689"),
)


def act_to_celex(label: str) -> Optional[str]:
    """First act whose keyword appears in ``label`` (a dataset act label or free text)."""
    low = label.lower()
    for keyword, celex in ACT_NAME_KEYWORDS:
        if keyword in low:
            return celex
    return None


def acts_mentioned_in(query: str) -> List[str]:
    """CELEX ids of every act named explicitly (by short name or number) in ``query``.

    Order-preserving and deduplicated. Used as a high-precision floor for act selection:
    an act the user names is a strong enough signal to include regardless of LLM scoring.
    """
    low = query.lower()
    found: List[str] = []
    for keyword, celex in ACT_NAME_KEYWORDS:
        if keyword in low and celex not in found:
            found.append(celex)
    return found
