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
    FactualCorrectness,
    Faithfulness,
)

from legal_assistant import config
from legal_assistant.rag.pipeline import RAGPipeline
from legal_assistant.logging_setup import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class RagasMetricsEvaluator:
    """Mirror of RagasEvaluatorUtils that scores RAG answers with the four
    standard RAGAS metrics instead of the custom precision/recall/faithfulness
    implementation:

    - faithfulness        (answer grounded in the retrieved contexts)
    - answer_relevancy    (answer relevant to the question)
    - context_precision   (retrieved contexts relevant, ranked correctly)
    - context_recall      (reference fully supported by retrieved contexts)
    - factual_precision   (fraction of answer claims supported by the reference, LLM-only)
    - factual_recall      (fraction of reference claims covered by the answer, LLM-only)
    """

    def __init__(
        self,
        method_id: str = "hybrid",
        use_context_curation: bool = False,
        use_query_decomposition: bool = False,
        synthesis_prompt_version: str | None = None,
    ):
        self.rag = RAGPipeline(
            method_id=method_id,
            use_context_curation=use_context_curation,
            use_query_decomposition=use_query_decomposition,
            synthesis_prompt_version=synthesis_prompt_version,
        )
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL or None)
        self.llm = llm_factory("gpt-4o-mini", client=self.client, max_tokens=16000)
        self.embeddings = OpenAIEmbeddings(client=self.client, model="text-embedding-3-small")

        self.faithfulness = Faithfulness(llm=self.llm)
        self.answer_relevancy = AnswerRelevancy(llm=self.llm, embeddings=self.embeddings)
        self.context_precision = ContextPrecision(llm=self.llm)
        self.context_recall = ContextRecall(llm=self.llm)
        # Precision and recall are logged separately instead of a single F1:
        # with the (untouchable) encyclopedic ground truths, F1 conflates two
        # very different signals: reference claims the corpus cannot support
        # (recall ceiling) and extra grounded claims in the answer (precision).
        self.factual_precision = FactualCorrectness(llm=self.llm, mode="precision", name="factual_precision")
        self.factual_recall = FactualCorrectness(llm=self.llm, mode="recall", name="factual_recall")

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
        (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            factual_precision,
            factual_recall,
        ) = await asyncio.gather(
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
            self.factual_precision.ascore(
                response=answer,
                reference=ground_truth,
            ),
            self.factual_recall.ascore(
                response=answer,
                reference=ground_truth,
            ),
        )

        scores = {
            "faithfulness": faithfulness.value,
            "answer_relevancy": answer_relevancy.value,
            "context_precision": context_precision.value,
            "context_recall": context_recall.value,
            "factual_precision": factual_precision.value,
            "factual_recall": factual_recall.value,
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
                "synthesis_prompt_version": evals.rag.synthesis_prompt_version,
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


METRIC_KEYS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "factual_precision",
    "factual_recall",
)


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


def main() -> None:
    global evals

    parser = argparse.ArgumentParser(description="Run the RAGAS evaluation experiment.")
    parser.add_argument(
        "--method", default="hybrid",
        help="RAG method id to evaluate (e.g. 'hybrid' or 'topics'). Default: hybrid.",
    )
    parser.add_argument(
        "--dataset", default="case_law_golden_dataset",
        help="Dataset name to load from the evals root dir.",
    )
    parser.add_argument(
        "--curate", action=argparse.BooleanOptionalAction, default=True,
        help="Enable/disable the pre-synthesis context-curation stage (default: on; use --no-curate to disable).",
    )
    parser.add_argument(
        "--decompose", action="store_true", default=False,
        help="Enable query decomposition (sub-questions + HyDE per sub-question) in the retriever.",
    )
    parser.add_argument(
        "--synthesis-prompt", default=None,
        help="Pin a registered answer_synthesis version (e.g. 'v9') instead of the active one.",
    )
    args = parser.parse_args()

    log_path = setup_run_logging()
    logger.info("Saving full run log to: %s", log_path)
    logger.info(
        "RAG method: %s | dataset: %s | curate: %s | decompose: %s | synthesis prompt: %s",
        args.method, args.dataset, args.curate, args.decompose,
        args.synthesis_prompt or "active",
    )
    evals = RagasMetricsEvaluator(
        method_id=args.method,
        use_context_curation=args.curate,
        use_query_decomposition=args.decompose,
        synthesis_prompt_version=args.synthesis_prompt,
    )
    try:
        asyncio.run(base_rag_experiment(
            dataset_name=args.dataset,
            root_dir=str(config.EVALS_DIR),
            output_path=str(
                config.EVALS_DIR / "evaluations"
                / f"rag_eval_ragas_{evals.rag.synthesis_prompt_version}_{uuid.uuid4()}.csv"
            ),
        ))
    finally:
        logger.info("Full run log saved to: %s", log_path.resolve())


if __name__ == "__main__":
    main()
