import json
import logging
import pathlib
import re
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from service.rag.methods.context import RagContext
from service.rag.methods.registry import get_method
from service.rag.prompt import (
    ANSWER_SYNTHESIS_PROMPT,
    ANSWER_FILTER_PROMPT,
    CONTEXT_CURATION_PROMPT,
    registry as prompt_registry,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

QA_PROMPT = PromptTemplate(
    template=ANSWER_SYNTHESIS_PROMPT,
    input_variables=["context", "question", "guidance"],
)

# First bracketed source header on a passage, e.g. "[Data Governance Act, ..., Article 3]".
_SOURCE_HEADER_RE = re.compile(r"^\s*(\[[^\]]+\])")


class RAGPipeline:

    def __init__(
        self,
        method_id: str = "hybrid",
        use_answer_filter: bool = False,
        use_context_curation: bool = False,
        use_query_decomposition: bool = True,
        hyde_iterations: int = 3,
        overrides: dict | None = None,
    ):
        """`overrides` sets any of the method's own params (see `method.param_specs()`),
        so an eval can A/B a single knob — e.g. `{"use_case_law": False}` — without a
        constructor argument per knob."""
        self.use_answer_filter = use_answer_filter
        self.use_context_curation = use_context_curation
        logger.info("[Prompts] active versions: %s", prompt_registry.active_versions())

        # Shared resources (graph, vector store, classifier, LLMs) come from the
        # same RagContext the chat frontend uses — single source of truth.
        ctx = RagContext()
        self.classifier = ctx.classifier
        self.synthesis_llm = ctx.synthesis_llm
        self.filter_llm = ctx.filter_llm if use_answer_filter else None
        self.curator_llm = ctx.curator_llm if use_context_curation else None

        method = get_method(method_id)
        config = method.default_config()
        if "hyde_iterations" in config:  # only the hybrid method consumes this
            config["hyde_iterations"] = hyde_iterations
        if "use_query_decomposition" in config:
            config["use_query_decomposition"] = use_query_decomposition
        for key, value in (overrides or {}).items():
            if key not in config:
                raise ValueError(f"{method_id!r} has no parameter {key!r}; known: {sorted(config)}")
            config[key] = value
        self.retriever = method.build_retriever(ctx, config)
        logger.info("[RAGPipeline] method=%s config=%s", method_id, config)

    def retrieve(self, question: str) -> dict:
        """Run only the retrieval step, without any LLM answer synthesis."""
        docs = self.retriever.invoke(question)
        return {
            "sources": [doc.metadata.get("id") for doc in docs],
            "contexts": [doc.page_content for doc in docs],
        }

    @staticmethod
    def _source_header(doc: Document) -> str:
        """The bracketed source header decorating a passage, or its id as fallback."""
        m = _SOURCE_HEADER_RE.match(doc.page_content)
        return m.group(1) if m else str(doc.metadata.get("id", ""))

    def _curate_context(
        self, question: str, docs: List[Document]
    ) -> Tuple[List[Document], str]:
        """Select the passages needed to answer the question with a cheap LLM.

        Filters (never rewrites) the retrieved passages, returning the kept ``Document``
        objects verbatim (governing-first) plus a ``guidance`` string naming the governing
        provision(s) for the synthesis prompt. Fail-open: on an empty selection or any
        parse error the full set is returned unchanged, since dropping a needed provision
        is unrecoverable downstream.
        """
        if not docs:
            return docs, ""

        numbered = "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs))
        prompt_text = CONTEXT_CURATION_PROMPT.format(
            question=question, numbered_passages=numbered
        )
        raw = self.curator_llm.invoke(prompt_text).content

        try:
            # Strip an optional ```json … ``` fence before parsing.
            cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", raw.strip()).strip()
            sel = json.loads(cleaned)
            keep = [i for i in sel["keep"] if isinstance(i, int) and 0 <= i < len(docs)]
            governing = {i for i in sel.get("governing", []) if i in keep}
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("[Curation] unparsable curator output — keeping all passages.")
            return docs, ""

        if not keep:  # fail-open on empty selection
            logger.warning("[Curation] empty selection — keeping all passages.")
            return docs, ""

        # Governing passages first, then the rest of the kept set — a positional signal
        # that reinforces the synthesis prompt's governing-provision step.
        ordered = [i for i in keep if i in governing] + [i for i in keep if i not in governing]
        kept_docs = [docs[i] for i in ordered]

        guidance = ""
        if governing:
            headers = ", ".join(self._source_header(docs[i]) for i in keep if i in governing)
            guidance = (
                f"The provision(s) that directly govern this question are: {headers}. "
                "Base the answer on them; treat the other passages only as cross-reference "
                "or exception support."
            )

        kept_ids = [
            f"{docs[i].metadata.get('id') or self._source_header(docs[i])}"
            f"{'(G)' if i in governing else ''}"
            for i in ordered
        ]
        dropped_ids = [
            str(docs[i].metadata.get("id") or self._source_header(docs[i]))
            for i in range(len(docs)) if i not in keep
        ]
        logger.info(
            "[Curation] kept %d/%d — kept: %s | dropped: %s  (G = governing)",
            len(kept_docs), len(docs), kept_ids, dropped_ids,
        )
        return kept_docs, guidance

    def query(self, question: str) -> dict:
        docs = self.retriever.invoke(question)

        guidance = ""
        if self.use_context_curation:
            docs, guidance = self._curate_context(question, docs)

        context = "\n\n".join(doc.page_content for doc in docs)
        prompt_text = QA_PROMPT.format(
            context=context, question=question, guidance=guidance,
        )
        answer_msg = self.synthesis_llm.invoke(prompt_text)
        answer = answer_msg.content.replace("\r\n", "\n").replace("\r", "\n").strip()

        if self.use_answer_filter:
            filtered = self.filter_llm.invoke(
                ANSWER_FILTER_PROMPT.format(question=question, draft_answer=answer)
            )
            answer = filtered.content.replace("\r\n", "\n").replace("\r", "\n").strip()

        return {
            "answer": answer,
            "sources": [doc.metadata.get("id") for doc in docs],
            "contexts": [doc.page_content for doc in docs],
        }

    def run_batch(self, questions_path: str, output_path: str) -> None:
        questions = json.loads(pathlib.Path(questions_path).read_text(encoding="utf-8"))
        output_file = pathlib.Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        report = []
        for i, item in enumerate(questions, 1):
            logger.info("Query %d/%d: %s", i, len(questions), item["question"])
            response = self.query(item["question"])
            report.append({
                "act": item["act"],
                "question": item["question"],
                "rag_response": response["answer"],
                "ground_truth": item["ground_truth"],
                "retrieved_context": response["sources"],
            })

        output_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Done. Results saved to: %s", output_file)



if __name__ == "__main__":
    rag = RAGPipeline()
    rag.run_batch("docs/question_recital_required.json", "results/rag_result.json")