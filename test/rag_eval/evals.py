import asyncio
import csv
import logging
import pathlib
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings
from openai import AsyncOpenAI
from ragas import Dataset, experiment, SingleTurnSample, DataTable
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics._answer_relevance import ResponseRelevancy
from ragas.metrics._context_precision import LLMContextPrecisionWithoutReference
from ragas.metrics.collections import ContextRelevance, Faithfulness

import config
from rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class RagasEvaluatorUtils:
    def __init__(self):
        self.rag = RAGPipeline()
        self.llm = llm_factory("gpt-4o-mini", client=AsyncOpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL or None), max_tokens=16000)
        self.embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            encode_kwargs={"normalize_embeddings": True},
        ))
        self.scorer = ContextRelevance(self.llm)

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

    async def calculate_metrics(self, question: str, retrieved_contexts: list[str], answer: str) -> dict:
        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=retrieved_contexts,
            response=answer,
        )
        faithfulness = await Faithfulness(llm=self.llm).ascore(
            user_input=question,
            response=answer,
            retrieved_contexts=retrieved_contexts,
        )
        response_relevancy = await ResponseRelevancy(llm=self.llm, embeddings=self.embeddings).single_turn_ascore(sample)
        context_precision = await LLMContextPrecisionWithoutReference(llm=self.llm).single_turn_ascore(sample)

        scores = {
            "faithfulness": faithfulness.value,
            "response_relevancy": response_relevancy,
            "context_precision": context_precision
        }
        logger.info("Scores: %s", scores)

        return scores

evals = RagasEvaluatorUtils()

@experiment()
async def base_rag_experiment(dataset_name: str, root_dir: str, output_path: str):
    dataset = evals.load_dataset_from_csv(dataset_name, root_dir)
    report = []

    for i, row in enumerate(dataset, 1):
        try:
            logger.info("Query %d/%d: %s", i, len(dataset), row["question"])
            response = evals.rag.query(row["question"])
            scores = await evals.calculate_metrics(row["question"], response["contexts"], response["answer"])
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

    return report


if __name__ == "__main__":
    asyncio.run(base_rag_experiment(
        dataset_name="golden_dataset",
        root_dir=str(config.EVALS_DIR),
        output_path=str(config.EVALS_DIR / "evaluations" / "rag_eval.csv"),
    ))


