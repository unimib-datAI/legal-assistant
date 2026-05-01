# Legal/structural stopwords that should never become topic labels
LEGAL_STOPWORDS = frozenset({
    # structural words from legislation
    "article", "paragraph", "section", "chapter", "regulation", "directive",
    "pursuant", "thereof", "herein", "whereas", "shall", "referred",
    "subparagraph", "annex", "recital", "provision", "point",
    # generic verbs/adjectives that leak through as topics
    "imposed", "processed", "implementing", "apply", "establish", "allow",
    "designate", "provided", "used", "approved", "issued", "empowered",
    "laid", "specified", "notified", "registered", "certain", "relevant",
    "available", "sufficiently", "proper", "balanced", "capable", "effective",
    # temporal/vague
    "week", "month", "year", "period", "later", "delay", "urgent", "while",
    "periodic", "receipt", "forum", "journal",
    # too generic
    "code", "activity", "document", "order", "rule", "level", "format",
    "amount", "process", "subject", "right", "relation", "standard",
    "regard", "purpose", "information", "country", "union", "market",
    "cost", "price", "staff", "power",
})

"""

NN: Nome, singolare (es. table, code)
NNS: Nome, plurale (es. tables, codes)
NNP: Nome proprio, singolare (es. Python, Italy)
VB / VBD / VBG: Verbi (forma base, passato, gerundio)
JJ: Aggettivo (es. green, large)
RB: Avverbio (es. very, silently
"""
# POS tags considered valid for topic terms (nouns and proper nouns)
VALID_POS_TAGS = frozenset({"NN", "NNS", "NNP", "NNPS"})