"""Provisions a CJEU judgment passage cites in its own text.

The graph's only case-law-to-act edge is document-level: ``(:CaseLaw)-[:INTERPRETS]->(:Article)``
says *this judgment interprets Article 58*, never *this passage does*. Ranking those by how
many of the retrieved judgments share an article rewards whatever every judgment happens to
touch (across the GDPR corpus the most-interpreted are Articles 6, 4, 82 and 5) rather than
what the passage in hand is actually about.

A judgment states which provisions it turns on by citing them, so read them off the text of
the passages retrieval already returned.

Matching is anchored on the ACT, not on the article number: a reference counts only when the
act is named immediately after the article list. Scanning for "Article N" and then guessing
the act does not survive real prose: Schrems II §199 reads

    "Article 1 of the Privacy Shield Decision is incompatible with Article 45(1) of the GDPR,
     read in the light of Articles 7, 8 and 47 of the Charter, and is therefore invalid."

where one sentence carries five article numbers and exactly one belongs to our corpus.

Every form an act is cited by is derived from its CELEX (see ``acts.celex_instrument_and_numbers``),
so an act added to the graph is matched here without touching this module.
"""
import logging
import re
from typing import List, Optional

from legal_assistant.rag.acts import CELEX_TO_ACT_NAME, celex_instrument_and_numbers

logger = logging.getLogger(__name__)

# The article number(s) of a citation: "45(1)", "56 and 60", "7, 8 and 47". Paragraph and
# point suffixes are captured only so they can be discarded: the act-side retrieval unit is
# the Article, so "Article 61(8)" must resolve to the same node as "Article 61".
#
# Ranges are deliberately NOT matched. "Articles 44 to 50 of the GDPR" names a scope, not a
# pinpoint, and expanding it would inject seven articles sharing one citing score, exactly
# the coarse signal this module exists to replace. The pattern fails on "to", so a range is
# skipped rather than mis-parsed.
_ITEM = r"\d+(?:\([^)]{1,16}\))*"
_ARTICLE_LIST = rf"(?P<nums>{_ITEM}(?:\s*(?:,|and)\s*{_ITEM})*)"

_PARENTHETICAL = re.compile(r"\([^)]*\)")
_NUMBER = re.compile(r"\d+")


def _act_reference(celex: str, resolve_anaphora: bool) -> Optional[str]:
    """Regex fragment matching every way a judgment names ``celex`` after an article list."""
    parsed = celex_instrument_and_numbers(celex)
    if parsed is None:
        return None
    word, numbers = parsed

    # "of Regulation (EU) 2016/679", "of Regulation No 2016/679", "of the Regulation 2016/679"
    joined = "|".join(re.escape(n) for n in numbers)
    alternatives = [rf"(?:the\s+)?{word}\s*(?:\([A-Z]{{2,3}}\)\s*)?(?:No\s*)?(?:{joined})"]

    # "of that regulation": the anaphora a judgment falls back to once it has cited the act
    # in full. Unambiguous only while a single act of this instrument type is in play, which
    # is the caller's business to know; hence the flag rather than a guess here.
    if resolve_anaphora:
        alternatives.append(rf"that\s+{word}")

    # "of the GDPR": the act's own short name, reusing the display map rather than adding a
    # second act-keyed table. An act cited only by number never matches this branch.
    name = CELEX_TO_ACT_NAME.get(celex)
    if name:
        alternatives.append(rf"the\s+{re.escape(name)}")

    return r"of\s+(?:" + "|".join(alternatives) + r")"


def _compile(celex: str, resolve_anaphora: bool) -> Optional[re.Pattern]:
    tail = _act_reference(celex, resolve_anaphora)
    if tail is None:
        return None
    return re.compile(rf"\bArticles?\s+{_ARTICLE_LIST}\s*{tail}", re.IGNORECASE)


def cited_articles(text: str, celex: str, *, resolve_anaphora: bool = True) -> List[str]:
    """Graph ids of the articles of ``celex`` that ``text`` explicitly cites.

    ``resolve_anaphora`` accepts the "of that regulation" shorthand. Pass ``False`` when more
    than one act of the same instrument type is in play, where "that regulation" is genuinely
    ambiguous and would resolve against every one of them.

    Returns ids in first-appearance order, deduplicated, e.g. ``["32016R0679art_58", ...]``.
    """
    pattern = _compile(celex, resolve_anaphora)
    if pattern is None:
        logger.debug("[Citations] %s is not a parsable CELEX, no citations resolved.", celex)
        return []

    found: List[str] = []
    for match in pattern.finditer(text):
        # Strip paragraph/point suffixes ("45(1)" -> "45") before reading the numbers, so
        # "Article 61(8)" and "Article 61" land on the same node.
        bare = _PARENTHETICAL.sub("", match.group("nums"))
        for number in _NUMBER.findall(bare):
            article_id = f"{celex}art_{int(number)}"
            if article_id not in found:
                found.append(article_id)
    return found
