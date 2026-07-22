"""Act-classification accuracy for the query classifier — classification only.

Unlike ``retrieval_eval.py`` / ``evals_ragas.py`` this runs neither retrieval nor answer
synthesis. For each golden-dataset row it calls ``QueryClassifier.classify`` and compares
the predicted target acts against the act the row is labelled with, then reports the
percentage of questions whose true act was detected.

Only the classifier LLM and the Neo4j graph (for the AVAILABLE ACTS block) are built — no
embedding model, vector store, or reranker — so a run is cheap relative to a full eval.

Metrics (the golden datasets are single-act, so the labelled act is the one true act):
- detected  — the labelled act is in the predicted set (the headline "% correct").
- exact     — the predicted set is exactly the labelled act, nothing extra.
- abstained — the classifier returned no acts (out-of-scope / below threshold).

Example:
    legal-assistant eval acts --dataset golden_dataset
    legal-assistant eval acts --dataset golden_dataset --threshold 0.5
    legal-assistant eval acts --dataset golden_dataset --limit 5
"""
import argparse
import csv
import logging
from collections import defaultdict
from datetime import datetime

from legal_assistant import config
from legal_assistant.logging_setup import configure_logging
from legal_assistant.rag.acts import CELEX_TO_ACT_NAME, act_to_celex
from legal_assistant.rag.intent_classifier import QueryClassifier
from legal_assistant.resources import make_chat_llm, make_langchain_graph

configure_logging()
logger = logging.getLogger(__name__)

_DATASETS_DIR = config.EVALS_DIR / "datasets"
_RESULTS_DIR = config.EVALS_DIR / "evaluations"


def _build_classifier(threshold: float) -> QueryClassifier:
    """Minimal classifier wiring (graph + LLM only), mirroring RagContext."""
    return QueryClassifier(
        graph=make_langchain_graph(),
        llm=make_chat_llm(),
        act_score_threshold=threshold,
    )


def _load_rows(dataset: str) -> list[dict]:
    path = _DATASETS_DIR / f"{dataset}.csv"
    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}")
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if rows and "act" not in rows[0]:
        raise SystemExit(
            f"Dataset {path.name} has no 'act' column (columns: {list(rows[0])}). "
            "This harness needs a single-act label per row."
        )
    return rows


def evaluate(dataset: str, threshold: float, limit: int | None) -> None:
    rows = _load_rows(dataset)
    if limit:
        rows = rows[:limit]
    classifier = _build_classifier(threshold)

    results: list[dict] = []
    detected = exact = abstained = scored = 0
    per_act_total: dict[str, int] = defaultdict(int)
    per_act_detected: dict[str, int] = defaultdict(int)

    for i, row in enumerate(rows, 1):
        question = (row.get("question") or "").strip()
        expected = act_to_celex(row.get("act") or "")
        if not question or expected is None:
            logger.warning("[%d] skipped (no question or unmapped act label %r)", i, row.get("act"))
            continue

        scored += 1
        per_act_total[expected] += 1
        predicted = classifier.classify(question).acts
        is_detected = expected in predicted
        is_exact = predicted == [expected]
        detected += is_detected
        exact += is_exact
        abstained += not predicted
        per_act_detected[expected] += is_detected

        mark = "OK " if is_detected else "XX "
        logger.info(
            "%s[%d/%d] exp=%s pred=%s | %s",
            mark, i, len(rows), expected, predicted, question[:70],
        )
        results.append({
            "question": question,
            "expected_celex": expected,
            "predicted_celex": "|".join(predicted),
            "detected": is_detected,
            "exact": is_exact,
        })

    if not scored:
        raise SystemExit("No scorable rows (every row was skipped).")

    logger.info("=" * 72)
    logger.info("Dataset: %s   threshold: %.2f   scored: %d rows", dataset, threshold, scored)
    logger.info("Act DETECTED (true act in prediction): %d/%d = %.1f%%", detected, scored, 100 * detected / scored)
    logger.info("EXACT (prediction == [true act]):      %d/%d = %.1f%%", exact, scored, 100 * exact / scored)
    logger.info("ABSTAINED (no act predicted):          %d/%d = %.1f%%", abstained, scored, 100 * abstained / scored)
    logger.info("-" * 72)
    for celex in sorted(per_act_total):
        tot, det = per_act_total[celex], per_act_detected[celex]
        logger.info("  %-22s %-11s %d/%d = %.1f%%", CELEX_TO_ACT_NAME.get(celex, celex), celex, det, tot, 100 * det / tot)
    logger.info("=" * 72)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = _RESULTS_DIR / f"act_classification_{dataset}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    logger.info("Per-row results written to %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="golden_dataset", help="CSV name under evals/datasets/ (no extension).")
    parser.add_argument("--threshold", type=float, default=config.ACT_SCORE_THRESHOLD,
                        help="Per-act relevance threshold for selecting an act (default: config.ACT_SCORE_THRESHOLD).")
    parser.add_argument("--limit", type=int, default=None, help="Classify only the first N rows (cheap smoke run).")
    args = parser.parse_args()
    evaluate(args.dataset, args.threshold, args.limit)


if __name__ == "__main__":
    main()
