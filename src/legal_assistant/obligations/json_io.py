"""Extract a JSON value from an LLM response.

The obligation prompts instruct the model to return only JSON, often fenced in triple
backticks. This adapts the source repository's ``extract_dict`` tolerance to this codebase,
kept small and dependency-free.
"""
from __future__ import annotations

import json
import re
from typing import Optional

_FENCED = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Optional[object]:
    """Parse the JSON object or array in ``text``, fenced or bare. None if there is none."""
    if not text:
        return None

    fenced = _FENCED.search(text)
    candidate = fenced.group(1).strip() if fenced else text.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return _first_bracketed(candidate)


def _first_bracketed(text: str) -> Optional[object]:
    """Fall back to the first balanced ``{...}`` or ``[...]`` span in the text."""
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None
