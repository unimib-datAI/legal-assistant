import argparse
import asyncio
import csv
import logging
import pathlib
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI
from ragas import Dataset, experiment, DataTable
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

import config
from rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class RagasMetricsEvaluator:
    """Mirror of RagasEvaluatorUtils that scores RAG answers with the four
    standard RAGAS metrics instead of the custom precision/recall/faithfulness
    implementation:

    - faithfulness        (answer grounded in the retrieved contexts)
    - answer_relevancy    (answer relevant to the question)
    - context_precision   (retrieved contexts relevant, ranked correctly)
    - context_recall      (reference fully supported by retrieved contexts)
    """

    def __init__(self, method_id: str = "hybrid"):
        self.rag = RAGPipeline(method_id=method_id)
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL or None)
        self.llm = llm_factory("gpt-4o-mini", client=self.client, max_tokens=16000)
        self.embeddings = OpenAIEmbeddings(client=self.client, model="text-embedding-3-small")

        self.faithfulness = Faithfulness(llm=self.llm)
        self.answer_relevancy = AnswerRelevancy(llm=self.llm, embeddings=self.embeddings)
        self.context_precision = ContextPrecision(llm=self.llm)
        self.context_recall = ContextRecall(llm=self.llm)

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

    async def calculate_metrics(
        self,
        question: str,
        retrieved_contexts: list[str],
        answer: str,
        ground_truth: str,
    ) -> dict:
        faithfulness, answer_relevancy, context_precision, context_recall = await asyncio.gather(
            self.faithfulness.ascore(
                user_input=question,
                response=answer,
                retrieved_contexts=retrieved_contexts,
            ),
            self.answer_relevancy.ascore(
                user_input=question,
                response=answer,
            ),
            self.context_precision.ascore(
                user_input=question,
                reference=ground_truth,
                retrieved_contexts=retrieved_contexts,
            ),
            self.context_recall.ascore(
                user_input=question,
                retrieved_contexts=retrieved_contexts,
                reference=ground_truth,
            ),
        )

        scores = {
            "faithfulness": faithfulness.value,
            "answer_relevancy": answer_relevancy.value,
            "context_precision": context_precision.value,
            "context_recall": context_recall.value,
        }
        logger.info("Scores: %s", scores)

        return scores


# Built in main() once the RAG method is chosen, so the heavy RagContext is not
# constructed at import time. base_rag_experiment resolves it as a global at call time.
evals: RagasMetricsEvaluator = None


@experiment()
async def base_rag_experiment(dataset_name: str, root_dir: str, output_path: str):
    dataset = evals.load_dataset_from_csv(dataset_name, root_dir)
    report = []

    for i, row in enumerate(dataset, 1):
        try:
            logger.info("Query %d/%d: %s", i, len(dataset), row["question"])
            response = evals.rag.query(row["question"])
            scores = await evals.calculate_metrics(
                row["question"], response["contexts"], response["answer"], row["ground_truth"]
            )
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

    log_metric_averages(report)

    return report


METRIC_KEYS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def log_metric_averages(report: list[dict]) -> None:
    if not report:
        logger.info("No rows to average.")
        return

    def _avg(rows: list[dict], key: str) -> float:
        values = [r[key] for r in rows if key in r and r[key] is not None]
        return sum(values) / len(values) if values else 0.0

    logger.info("=== Overall averages (n=%d) ===", len(report))
    for key in METRIC_KEYS:
        logger.info("  %-18s %.3f", key, _avg(report, key))

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
    which is a no-op here because importing rag_pipeline already configured the
    root logger). Returns the log path so the run can report where it was saved.
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


def main() -> None:
    global evals

    parser = argparse.ArgumentParser(description="Run the RAGAS evaluation experiment.")
    parser.add_argument(
        "--method", default="hybrid",
        help="RAG method id to evaluate (e.g. 'hybrid' or 'topics'). Default: hybrid.",
    )
    parser.add_argument(
        "--dataset", default="subset_retrieval_scarso",
        help="Dataset name to load from the evals root dir.",
    )
    args = parser.parse_args()

    log_path = setup_run_logging()
    logger.info("Saving full run log to: %s", log_path)
    logger.info("RAG method: %s | dataset: %s", args.method, args.dataset)
    evals = RagasMetricsEvaluator(method_id=args.method)
    try:
        asyncio.run(base_rag_experiment(
            dataset_name=args.dataset,
            root_dir=str(config.EVALS_DIR),
            output_path=str(config.EVALS_DIR / "evaluations" / f"rag_eval_ragas_{uuid.uuid4()}.csv"),
        ))
    finally:
        logger.info("Full run log saved to: %s", log_path.resolve())


if __name__ == "__main__":
    main()
