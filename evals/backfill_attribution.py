"""Add per-sentence source attribution to an existing evaluation CSV.

Attribution needs only the answer text and the passages that produced it, both are
already stored in the results CSV (``answer``, ``sources``, ``contexts``). So an
existing run can be enriched *post hoc*, without re-running retrieval, synthesis or
the RAGAS metrics: one LLM call per row, nothing else.

Two columns are appended, both JSON:

``segments``       ``[{"text": "...", "markers": ["S1"]}, ...]``
``cited_sources``  ``[{"marker": "S1", "id": "32022R0868art_1"}, ...]``

The answer is never rewritten: sentences are split locally and the LLM only returns
marker assignments (see :mod:`legal_assistant.rag.attribution`).

Usage::

    legal-assistant eval backfill --csv .../best_bench.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.language_models import BaseChatModel

from legal_assistant import config
from legal_assistant.logging_setup import configure_logging
from legal_assistant.rag.attribution import SourceRef, attribute_answer, keep_cited_sources
from legal_assistant.resources import make_chat_llm

configure_logging()
logger = logging.getLogger(__name__)

# Passages are decorated at retrieval time with a bracketed header, e.g.
# "[Data Governance Act, Chapter I — General provisions, Subject matter and scope]".
_HEADER_RE = re.compile(r"^\s*\[([^\]]+)\]")


def _parse_header(context: str) -> tuple[str, str]:
    """Pull (act, title) out of a passage's bracketed header; ("", "") when absent."""
    match = _HEADER_RE.match(context)
    if not match:
        return "", ""
    parts = [p.strip() for p in match.group(1).split(",")]
    act = parts[0] if parts else ""
    title = parts[-1] if len(parts) > 1 else ""
    return act, title


def _build_source_refs(source_ids: List[str], contexts: List[str]) -> List[SourceRef]:
    """Rebuild the S1..Sn numbered passages the answer was synthesised from."""
    refs = []
    for i, (doc_id, context) in enumerate(zip(source_ids, contexts), 1):
        act, title = _parse_header(context)
        refs.append(SourceRef(
            marker=f"S{i}",
            doc_id=doc_id,
            act=act,
            title=title,
            type="recital" if "rct_" in doc_id else "article",
            text=context,
        ))
    return refs


def _split_pipe(value: object) -> List[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def backfill(csv_path: Path, llm: BaseChatModel) -> pd.DataFrame:
    """Return the CSV's frame with `segments` and `cited_sources` columns added."""
    df = pd.read_csv(csv_path)
    missing = {"answer", "sources", "contexts"} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} lacks required column(s): {sorted(missing)}")

    all_segments, all_cited = [], []
    for i, row in enumerate(df.itertuples(index=False), 1):
        source_ids = _split_pipe(row.sources)
        contexts = str(row.contexts).split("|") if not pd.isna(row.contexts) else []

        if not source_ids or len(source_ids) != len(contexts):
            # No passages recorded (or a malformed row): nothing to attribute against.
            logger.warning("Row %d: %d source(s), %d context(s), skipped",
                           i, len(source_ids), len(contexts))
            all_segments.append("")
            all_cited.append("")
            continue

        refs = _build_source_refs(source_ids, contexts)
        segments = attribute_answer(str(row.answer), refs, llm)
        segments, cited = keep_cited_sources(segments, refs)

        tagged = sum(1 for seg in segments if seg.source_markers)
        logger.info("Row %d/%d: %d/%d sentence(s) attributed, %d/%d source(s) cited",
                    i, len(df), tagged, len(segments), len(cited), len(refs))

        all_segments.append(json.dumps(
            [{"text": s.text, "markers": s.source_markers} for s in segments],
            ensure_ascii=False,
        ))
        all_cited.append(json.dumps(
            [{"marker": r.marker, "id": r.doc_id} for r in cited], ensure_ascii=False,
        ))

    df["segments"] = all_segments
    df["cited_sources"] = all_cited
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Evaluation results CSV to enrich.")
    parser.add_argument("--out", help="Output path (default: overwrite --csv in place).")
    parser.add_argument("--model", default=config.RAG_LLM_MODEL,
                        help="Chat model used for the attribution pass.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        logger.error("No such file: %s", csv_path)
        return 1

    df = backfill(csv_path, make_chat_llm(model=args.model))

    out_path = Path(args.out) if args.out else csv_path
    df.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Wrote %d row(s) with attribution to %s", len(df), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
