import asyncio
import csv
import logging
import pathlib
from collections import defaultdict
from datetime import datetime
from typing import Any
import uuid

from openai import AsyncOpenAI
from ragas import Dataset, experiment, DataTable
from ragas.llms import llm_factory
from ragas.metrics.collections import FactualCorrectness, Faithfulness

from legal_assistant import config
from legal_assistant.rag.pipeline import RAGPipeline
from legal_assistant.logging_setup import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class RagasEvaluatorUtils:
    def __init__(self):
        self.rag = RAGPipeline()
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL or None)
        self.llm = llm_factory("gpt-4o-mini", client=self.client, max_tokens=16000)

    def load_dataset_from_csv(self, name: str, root_dir: str) -> DataTable[Any]:
        return Dataset.load(name=name, backend="local/csv", root_dir=root_dir)

    def create_csv_report(self, report: list[dict], output_path: str) -> pathlib.Path:
        path = pathlib.Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=report[0].keys())
            writer.writeheader()
            writer.writerows(report)
        return path

    async def extract_claims(self, text: str, atomicity: str = "low", coverage: str = "low") -> list[str]:
        scorer = FactualCorrectness(llm=self.llm, atomicity=atomicity, coverage=coverage)
        return await scorer._decompose_claims(text)

    async def evaluate_precision_recall_f1(
        self,
        response_claims: list[str],
        reference_claims: list[str],
        response_text: str,
        reference_text: str,
    ) -> tuple[dict, dict]:
        scorer = FactualCorrectness(llm=self.llm)

        resp_ref, ref_resp = await asyncio.gather(
            scorer._verify_claims(response_claims, reference_text),
            scorer._verify_claims(reference_claims, response_text),
        )
        tp = sum(v.verdict for v in resp_ref.statements)
        fp = len(resp_ref.statements) - tp
        fn = sum(not v.verdict for v in ref_resp.statements)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        verdicts_dict = {
            "response_vs_reference": [
                {"statement": v.statement, "verdict": int(v.verdict), "reason": v.reason}
                for v in resp_ref.statements
            ],
            "reference_vs_response": None
            if ref_resp is None
            else [
                {"statement": v.statement, "verdict": int(v.verdict), "reason": v.reason}
                for v in ref_resp.statements
            ],
        }

        return metrics, verdicts_dict

    async def evaluate_faithfulness_multi(self, statements: list[str], context: list[str]) -> tuple[float, list[dict]]:
        """
        Evaluate faithfulness over a corpus by checking each document independently.
        A statement is supported if at least one document supports it.
        """
        scorer = Faithfulness(llm=self.llm)
        per_statement = defaultdict(list)

        all_verdicts = await asyncio.gather(
            *[scorer._create_verdicts(statements, doc) for doc in context]
        )
        for verdicts in all_verdicts:
            for v in verdicts.statements:
                per_statement[v.statement].append({
                    "verdict": int(v.verdict),
                    "reason": v.reason,
                })

        aggregated_verdicts = []
        for statement, verdicts in per_statement.items():
            supporting = [v for v in verdicts if v["verdict"] == 1]
            non_supporting = [v for v in verdicts if v["verdict"] == 0]

            if supporting:
                aggregated_verdicts.append({
                    "statement": statement,
                    "verdict": 1,
                    "reason": supporting[0]["reason"],
                })
            else:
                aggregated_verdicts.append({
                    "statement": statement,
                    "verdict": 0,
                    "reason": [v["reason"] for v in non_supporting],
                })

        supported = sum(v["verdict"] for v in aggregated_verdicts)
        total = len(aggregated_verdicts)
        score = supported / total if total > 0 else 0.0

        return score, aggregated_verdicts

    async def calculate_metrics(
        self,
        retrieved_contexts: list[str],
        answer: str,
        ground_truth: str,
    ) -> dict:
        reference_claims, response_claims = await asyncio.gather(
            self.extract_claims(ground_truth),
            self.extract_claims(answer),
        )

        logger.info("[Claims] Ground truth (%d): %s", len(reference_claims), reference_claims)
        logger.info("[Claims] Response     (%d): %s", len(response_claims), response_claims)

        (pr_metrics, verdicts), (faith_score, _) = await asyncio.gather(
            self.evaluate_precision_recall_f1(response_claims, reference_claims, answer, ground_truth),
            self.evaluate_faithfulness_multi(response_claims, retrieved_contexts),
        )

        for v in verdicts["response_vs_reference"]:
            label = "TP" if v["verdict"] else "FP"
            logger.info("[%s] response claim: %s | reason: %s", label, v["statement"], v["reason"])
        if verdicts["reference_vs_response"]:
            for v in verdicts["reference_vs_response"]:
                if not v["verdict"]:
                    logger.info("[FN] missed GT claim: %s | reason: %s", v["statement"], v["reason"])

        scores = {
            "tp": pr_metrics["tp"],
            "fp": pr_metrics["fp"],
            "fn": pr_metrics["fn"],
            "precision": pr_metrics["precision"],
            "recall": pr_metrics["recall"],
            "f1": pr_metrics["f1"],
            "faithfulness": faith_score,
        }
        logger.info("Scores: %s", scores)

        return scores

evals = RagasEvaluatorUtils()

# Set to True to test retrieval only: skips LLM answer synthesis and metric
# computation (both require LLM calls). Flip back to False for full evaluation.
RETRIEVAL_ONLY = False


@experiment()
async def base_rag_experiment(dataset_name: str, root_dir: str, output_path: str):
    dataset = evals.load_dataset_from_csv(dataset_name, root_dir)
    report = []

    for i, row in enumerate(dataset, 1):
        try:
            logger.info("Query %d/%d: %s", i, len(dataset), row["question"])

            if RETRIEVAL_ONLY:
                response = evals.rag.retrieve(row["question"])
                report.append({
                    **row,
                    "sources": "|".join(response["sources"]),
                    "contexts": "|".join(response["contexts"]),
                })
                continue

            response = evals.rag.query(row["question"])
            scores = await evals.calculate_metrics(response["contexts"], response["answer"], row["ground_truth"])
            report.append({
                **row,
                "answer": response["answer"],
                "sources": "|".join(response["sources"]),
                "contexts": "|".join(response["contexts"]),
                **scores,
            })
        except Exception as e:
            logger.error("Row %d failed: %s", i, e, exc_info=True)

    result_path = evals.create_csv_report(report, output_path)
    logger.info("Results saved to: %s", result_path)

    if not RETRIEVAL_ONLY:
        log_metric_averages(report)

    return report


METRIC_KEYS = ("precision", "recall", "f1", "faithfulness")


def log_metric_averages(report: list[dict]) -> None:
    if not report:
        logger.info("No rows to average.")
        return

    def _avg(rows: list[dict], key: str) -> float:
        values = [r[key] for r in rows if key in r and r[key] is not None]
        return sum(values) / len(values) if values else 0.0

    logger.info("=== Overall averages (n=%d) ===", len(report))
    for key in METRIC_KEYS:
        logger.info("  %-13s %.3f", key, _avg(report, key))

    by_act = defaultdict(list)
    for row in report:
        by_act[row.get("act", "unknown")].append(row)

    logger.info("=== Per-act averages ===")
    for act, rows in sorted(by_act.items()):
        scores = "  ".join(f"{key}={_avg(rows, key):.3f}" for key in METRIC_KEYS)
        logger.info("  [%s] (n=%d)  %s", act, len(rows), scores)


def setup_run_logging() -> pathlib.Path:
    """Tee all INFO logs of this run to a timestamped file under evals/logs.

    A FileHandler is attached explicitly (rather than via logging.basicConfig,
    which is a no-op once configure_logging() has run). Returns the log path so the run can report where it was saved.
    """
    log_dir = config.EVALS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"eval_run_{datetime.now():%Y%m%d_%H%M%S}.log"

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    return log_path


if __name__ == "__main__":
    log_path = setup_run_logging()
    logger.info("Saving full run log to: %s", log_path)
    try:
        asyncio.run(base_rag_experiment(
            dataset_name="subset_retrieval_scarso",
            root_dir=str(config.EVALS_DIR),
            output_path=str(config.EVALS_DIR / "evaluations" / f"rag_eval_{uuid.uuid4()}.csv"),
        ))
    finally:
        logger.info("Full run log saved to: %s", log_path.resolve())


