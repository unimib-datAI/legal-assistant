"""Single source of truth for the four EU acts: CELEX ids, display names, and the
surface keyword lexicon used to recognise an act by name or regulation number.

Consumers:
- ``rag/documents.py``: CELEX -> display name for passage headers.
- ``rag/intent_classifier.py``: ``acts_mentioned_in`` as a deterministic floor, an act the
  user names explicitly is force-included regardless of the LLM's relevance score.
- ``evals/retrieval_eval.py``: ``act_to_celex`` maps a dataset act label to CELEX.
"""
import re
from typing import List, Optional, Tuple

# CELEX is structured: sector, year, instrument letter, number. 32016R0679 is sector 3,
# year 2016, Regulation, number 679. Everything an act is cited BY inside a judgment is
# therefore derivable from its id, so a new act needs no entry anywhere below.
_CELEX_RE = re.compile(r"^(?P<sector>\d)(?P<year>\d{4})(?P<instrument>[A-Z]+)(?P<number>\d+)$")

# Keyed by the CELEX instrument letter, not by any act: every Regulation added to the graph
# gets "Regulation" for free.
_INSTRUMENT_WORD = {
    "R": "Regulation",
    "L": "Directive",
    "D": "Decision",
}

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


def celex_instrument_and_numbers(celex: str) -> Optional[Tuple[str, List[str]]]:
    """The instrument word and the number form(s) a judgment cites ``celex`` by.

    ``32016R0679`` -> ``("Regulation", ["2016/679"])``. Both are read off the CELEX itself,
    so this holds for any act in the graph rather than a hardcoded four.

    Two number forms are returned for pre-2000 instruments: EU numbering abbreviated the
    year until then, so Directive 95/46 (``31995L0046``) is cited "95/46", never "1995/46".
    """
    m = _CELEX_RE.match(celex.strip().upper())
    if not m:
        return None
    word = _INSTRUMENT_WORD.get(m.group("instrument"))
    if not word:
        return None

    year, number = int(m.group("year")), int(m.group("number"))
    forms = [f"{year}/{number}"]
    if year < 2000:
        forms.append(f"{year % 100:02d}/{number}")
    return word, forms


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
