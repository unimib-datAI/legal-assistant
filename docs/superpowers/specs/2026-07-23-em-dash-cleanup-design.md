# Em dash cleanup in comments and docstrings

**Date:** 2026-07-23
**Status:** approved, pending implementation

## Problem

The character `—` (U+2014, em dash) appears 368 times across the repository. Most
occurrences are stylistic punctuation in comments and docstrings, introduced by
generated code. The goal is twofold: remove it where it is merely punctuation, and
record a rule so it does not come back.

The character is not uniformly removable. Three distinct zones exist, and conflating
them would cause regressions.

| Zone | Where | Nature | Action |
|---|---|---|---|
| Prose | comments, docstrings, log messages | punctuation | rewrite |
| Prompt contract | `rag/prompts/`, `rag/documents.py` | LLM-visible token | leave intact |
| Parsed data | `case_law/html_parser.py` | matches real EUR-Lex text | leave intact |

The prompt contract is the sharpest constraint. `rag/documents.py` builds the literal
source header `[{act_name}, Chapter {chapter} — {chapter_title}, {title}]`, and the
synthesis prompts instruct the model to cite from exactly that shape. The two are
coupled: changing one without the other breaks citation. Changing both is coherent but
alters prompt text, which invalidates existing eval baselines. Neither is in scope here.

The parsed-data zone is narrower but equally firm. `_DASH = r"[-–—]"` in
`case_law/html_parser.py` matches em dashes present in the EUR-Lex source markup, and
the neighbouring comments reproduce authentic source strings as documentation.

The en dash `–` (U+2013) needs no treatment: its three occurrences fall entirely inside
the excluded zones.

## Scope

**In:** `.py` files under `src/`, `tests/`, `frontend/`, `evals/`.

**Out:** `docs/`, `README.md`, `corpus/`, `venv/`, and the whole of
`src/legal_assistant/rag/prompts/`.

`rag/prompts/` is excluded in full, docstrings included. The folder is content sent to
the model, and a blanket exclusion keeps the boundary unambiguous.

## Method

Edits target only tokens the Python parser classifies as prose:

- `tokenize` identifies `COMMENT` tokens.
- `ast` collects module, class, and function docstrings.
- Log-message string literals are handled as a third, separately reviewed group.

A line-oriented regex is deliberately not used. It cannot distinguish
`# arithmetic — no query needed` from `_DASH = r"[-–—]"`; the parser can.

Log messages are string literals, so the "comments and docstrings" criterion would
exclude them. They are included because none is functional — each is diagnostic prose,
for example `logger.debug("[Citations] %s is not a parsable CELEX — no citations
resolved.", celex)`. They are grouped separately in the diff so they can be assessed at
a glance.

## Replacement

Rewritten case by case rather than substituted mechanically:

- colon where the second clause explains the first
- comma for parenthetical asides
- full stop where the clause stands alone
- plain hyphen in tabular headings, where the character is a separator rather than
  punctuation: `# LAYER 3 — GDPR DATA SUBJECT RIGHTS` becomes
  `# LAYER 3 - GDPR DATA SUBJECT RIGHTS`

## Verification

1. `pytest` stays green.
2. `git diff` contains no modified line outside a comment, a docstring, or a log message.
3. A final grep confirms the surviving occurrences are exactly those in `rag/prompts/`,
   `rag/documents.py`, and `case_law/html_parser.py`.

## Durable rule

A bullet under *Style & Structure* in `.claude/CLAUDE.md`: no em dash in comments,
docstrings, or log messages, with the exception stated for prompt text and for regexes
matching EUR-Lex source, where the character is part of the data.
