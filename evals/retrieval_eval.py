"""Deterministic retrieval-quality metric for the RAG methods.

Unlike ``evals_ragas.py`` (which scores the *generated answer* with an LLM judge,
noisy at small n), this harness scores only the *retrieval* step against the
articles/recitals that each ground-truth answer cites inline. It calls
``RAGPipeline.retrieve`` (no answer synthesis, no judge), so a run is cheap and,
apart from HyDE sampling, deterministic: the right tool for A/B-ing reranker
choices (the cross-encoder set via the RERANK_MODEL env var) without generation/judge noise.

Metrics per query, aggregated overall and per act (rows whose GT cites no
target-act provision are skipped from the aggregates):

- top1_anchor:  is the top-ranked retrieved doc one the GT relies on?
- recall_at_k:  fraction of the GT's cited provisions present in the retrieved set
- mrr:          1 / rank of the first GT-cited provision in the retrieved list

Example:
    legal-assistant eval retrieval --dataset subset_retrieval_scarso --repeats 3
    RERANK_MODEL=Alibaba-NLP/gte-reranker-modernbert-base \
        legal-assistant eval retrieval --dataset subset_retrieval_scarso --repeats 3
"""
import argparse
import csv
import logging
import pathlib
import re
import uuid
from collections import defaultdict
from datetime import datetime

from legal_assistant import config
from legal_assistant.rag.pipeline import RAGPipeline
from legal_assistant.rag.acts import act_to_celex
from legal_assistant.logging_setup import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# Dataset ``act`` label -> CELEX is handled by ``act_to_celex`` (legal_assistant/rag/acts.py),
# which tolerates label variants ("Data Governance Act 2022/868", "GDPR", "Data Act 2023/2854").

# References to provisions of the TARGET act. Cross-references to other instruments
# ("Article 4, point (1), of Regulation (EU) 2016/679") are dropped: the tell is a
# numbered/parenthesised Regulation or a Directive within a short window after the ref.
_ART_RE = re.compile(r"\bArticles?\s+(\d+)", re.IGNORECASE)
_RCT_RE = re.compile(r"\bRecitals?\s+(\d+)", re.IGNORECASE)
_CROSSREF_RE = re.compile(r"\bof\s+(that\s+|the\s+)?(Directive|Regulation\s*(\(E[UC]\)|No|\d))", re.IGNORECASE)


def _numbered_refs(text: str, pattern: re.Pattern) -> set:
    refs = set()
    for match in pattern.finditer(text):
        window = text[match.end(): match.end() + 40]
        if _CROSSREF_RE.match(window.lstrip(" ,")):
            continue  # cross-reference to another regulation/directive
        refs.add(int(match.group(1)))
    return refs


def expected_ids(ground_truth: str, celex: str) -> set:
    """The set of provision IDs (``{celex}art_N`` / ``{celex}rct_N``) the GT cites."""
    articles = {f"{celex}art_{n}" for n in _numbered_refs(ground_truth, _ART_RE)}
    recitals = {f"{celex}rct_{n}" for n in _numbered_refs(ground_truth, _RCT_RE)}
    return articles | recitals


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def score_query(sources: list, expected: set) -> dict:
    """One retrieval's metrics against the expected provision set."""
    retrieved = set(sources)
    top1 = 1.0 if sources and sources[0] in expected else 0.0
    recall = len(expected & retrieved) / len(expected) if expected else 0.0
    mrr = 0.0
    for rank, source_id in enumerate(sources, 1):
        if source_id in expected:
            mrr = 1.0 / rank
            break
    return {"top1_anchor": top1, "recall_at_k": recall, "mrr": mrr}


METRIC_KEYS = ("top1_anchor", "recall_at_k", "mrr")


def evaluate(rag: RAGPipeline, rows: list, repeats: int) -> list:
    report = []
    for i, row in enumerate(rows, 1):
        celex = act_to_celex(row["act"])
        expected = expected_ids(row["ground_truth"], celex) if celex else set()

        runs = defaultdict(list)
        last_sources: list = []
        for _ in range(repeats):
            last_sources = rag.retrieve(row["question"])["sources"]
            scores = score_query(last_sources, expected)
            for key, value in scores.items():
                runs[key].append(value)

        record = {
            "act": row["act"],
            "question": row["question"],
            "n_expected": len(expected),
            "expected": "|".join(sorted(expected)),
            "sources": "|".join(last_sources),
            **{key: _mean(runs[key]) for key in METRIC_KEYS},
        }
        report.append(record)
        logger.info(
            "Q%d/%d [n_exp=%d] top1=%.2f recall@k=%.2f mrr=%.2f | %s",
            i, len(rows), len(expected),
            record["top1_anchor"], record["recall_at_k"], record["mrr"],
            row["question"][:60],
        )
    return report


def log_averages(report: list) -> None:
    scored = [r for r in report if r["n_expected"] > 0]
    skipped = len(report) - len(scored)
    if not scored:
        logger.warning("No rows had extractable GT references, nothing to average.")
        return

    logger.info("=== Overall (n=%d scored, %d skipped: GT cited no target-act provision) ===",
                len(scored), skipped)
    for key in METRIC_KEYS:
        logger.info("  %-12s %.3f", key, _mean([r[key] for r in scored]))

    by_act = defaultdict(list)
    for row in scored:
        by_act[row["act"]].append(row)
    logger.info("=== Per-act ===")
    for act, rows in sorted(by_act.items()):
        stats = "  ".join(f"{key}={_mean([r[key] for r in rows]):.3f}" for key in METRIC_KEYS)
        logger.info("  [%s] (n=%d)  %s", act, len(rows), stats)


def load_dataset(name: str) -> list:
    path = config.EVALS_DIR / "datasets" / f"{name}.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(report: list, output_path: pathlib.Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic retrieval-quality eval.")
    parser.add_argument("--method", default="hybrid", help="RAG method id (default: hybrid).")
    parser.add_argument("--dataset", default="subset_retrieval_scarso",
                        help="Dataset name under evals/datasets/ (without .csv).")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Runs per query, averaged to wash out HyDE sampling variance.")
    parser.add_argument("--no-case-law", action="store_true",
                        help="Disable the INTERPRETIVE case law branch and its graph boost; "
                             "the control arm when A/B-ing case law retrieval.")
    args = parser.parse_args()

    overrides = {"use_case_law": False} if args.no_case_law else None
    rag = RAGPipeline(method_id=args.method, overrides=overrides)

    rows = load_dataset(args.dataset)
    logger.info("[retrieval_eval] method=%s dataset=%s rows=%d repeats=%d case_law=%s",
                args.method, args.dataset, len(rows), args.repeats, not args.no_case_law)

    report = evaluate(rag, rows, args.repeats)

    tag = config.RERANK_MODEL.rsplit("/", 1)[-1]
    output_path = (config.EVALS_DIR / "evaluations" /
                   f"retrieval_eval_{tag}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.csv")
    write_csv(report, output_path)
    logger.info("Results saved to: %s", output_path)

    log_averages(report)


if __name__ == "__main__":
    main()
