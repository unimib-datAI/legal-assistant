"""Runtime engine that runs a chosen RAG method and produces a prose answer.

The engine retrieves passages with the chosen method, synthesises a continuous
legal-prose answer with the synthesis LLM, and lists the retrieved passages as
the answer's sources.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from service.rag.attribution import attribute_answer, keep_cited_sources
from service.rag.methods.base import AttributedAnswer, Segment, SourceRef
from service.rag.methods.context import RagContext
from service.rag.methods.registry import REGISTRY
from service.rag.prompt import ANSWER_SYNTHESIS_PROMPT
from service.rag.rag_alternative import _CELEX_TO_ACT_NAME, _doc_id

logger = logging.getLogger(__name__)

_SYNTHESIS_PROMPT = PromptTemplate(
    template=ANSWER_SYNTHESIS_PROMPT,
    input_variables=["context", "question", "guidance"],
)


def _celex_of(doc_id: str) -> str:
    """A doc id like '32016R0679_art_5' starts with its act's CELEX."""
    return doc_id.split("_", 1)[0] if doc_id else ""


def _build_sources(docs: List[Document]) -> Tuple[List[SourceRef], Dict[str, SourceRef]]:
    """Number passages with [Sn] markers and build SourceRefs + a lookup."""
    sources: List[SourceRef] = []
    by_marker: Dict[str, SourceRef] = {}
    for i, doc in enumerate(docs, 1):
        marker = f"S{i}"
        meta = doc.metadata
        dtype = meta.get("type", "article")
        doc_id = _doc_id(doc)

        if dtype == "recital":
            celex = meta.get("celex") or _celex_of(doc_id)
            number = meta.get("number")
            title = f"Recital {number}" if number is not None else "Recital"
        else:
            celex = meta.get("act") or _celex_of(doc_id)
            title = meta.get("title") or meta.get("article_title") or ""

        ref = SourceRef(
            marker=marker,
            doc_id=doc_id,
            act=_CELEX_TO_ACT_NAME.get(celex, celex),
            title=title,
            type=dtype,
            text=doc.page_content,
        )
        sources.append(ref)
        by_marker[marker] = ref
    return sources, by_marker


class RagEngine:
    """Runs RAG methods from the registry and returns prose answers."""

    def __init__(self, ctx: RagContext) -> None:
        self.ctx = ctx

    def answer(self, method_id: str, question: str, config: Dict[str, Any]) -> AttributedAnswer:
        method = REGISTRY[method_id]
        logger.info("[RagEngine] method=%s config=%s", method_id, config)

        retriever = method.build_retriever(self.ctx, config)
        docs = retriever.invoke(question)
        if not docs:
            logger.warning("[RagEngine] No documents retrieved.")
            return AttributedAnswer(
                segments=[Segment(text="No relevant passages were retrieved for this question.")],
                sources=[],
                raw_answer="No relevant passages were retrieved for this question.",
            )

        sources, _ = _build_sources(docs)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt_text = _SYNTHESIS_PROMPT.format(
            context=context, question=question, guidance="",
        )
        answer_msg = self.ctx.synthesis_llm.invoke(prompt_text)
        answer_text = answer_msg.content.replace("\r\n", "\n").replace("\r", "\n").strip()

        logger.info("[RagEngine] synthesised answer from %d passage(s)", len(sources))
        segments = attribute_answer(answer_text, sources, self.ctx.synthesis_llm)
        segments, sources = keep_cited_sources(segments, sources)
        return AttributedAnswer(
            segments=segments,
            sources=sources,
            raw_answer=answer_text,
        )
