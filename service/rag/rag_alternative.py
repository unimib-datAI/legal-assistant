import logging
import re
from typing import List, Any, Optional

import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from sentence_transformers.cross_encoder import CrossEncoder

from service.graph.query import NodeQueries

logger = logging.getLogger(__name__)

_CELEX_TO_ACT_NAME = {
    "32022R0868": "Data Governance Act",
    "32023R2854": "Data Act",
    "32024R1689": "AI Act",
    "32016R0679": "GDPR",
}

_DISPLAY_NUM_RE = re.compile(r"^\((\d+)\)\s*")


def _recital_header(celex: str, recital_text: str) -> str:
    """Build a '[Act, Recital N]' prefix using the display label parsed from the recital text."""
    act_name = _CELEX_TO_ACT_NAME.get(celex, celex)
    m = _DISPLAY_NUM_RE.match(recital_text)
    if m:
        return f"[{act_name}, Recital {m.group(1)}]"
    return f"[{act_name}, Recital]"


class ArticleTraversalRetriever(BaseRetriever):
    """Retriever that fetches recitals via BM25 and re-ranks them with a cross-encoder."""

    graph: Any
    k: int = 10
    top_k_recitals: int = 20
    cross_encoder: Any = CrossEncoder("BAAI/bge-reranker-v2-m3")
    classifier: Any = None
    _recital_cache: Optional[dict] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _load_recitals(self, acts: List[str]) -> dict:
        """Fetch all recitals for the given acts and build a BM25 index. Results are cached."""
        cache_key = tuple(sorted(acts))

        if self._recital_cache is not None and cache_key in self._recital_cache:
            return self._recital_cache[cache_key]

        results = self.graph.query(
            NodeQueries.GET_RECITALS_BY_ACTS,
            params={"acts": acts}
        )

        if not results:
            data = {"bm25": None}
        else:
            recital_docs = [
                Document(
                    page_content=r["text"],
                    metadata={"id": r["recital_id"], "celex": r["celex"]},
                )
                for r in results
            ]
            bm25 = BM25Retriever.from_documents(recital_docs)
            data = {"bm25": bm25}
            logger.info(
                "[Recital Cache] Loaded %d recitals + built BM25 index for acts %s",
                len(recital_docs), acts,
            )

        if self._recital_cache is None:
            self._recital_cache = {}
        self._recital_cache[cache_key] = data
        return data

    def _match_user_query_to_recitals(
        self, user_query: str, target_acts: List[str]
    ) -> List[Document]:
        """Return top-k recitals by BM25 lexical match against the user query."""
        data = self._load_recitals(target_acts)
        bm25 = data.get("bm25")
        if bm25 is None:
            return []

        bm25.k = self.top_k_recitals
        results = bm25.invoke(user_query)
        if results:
            logger.info(
                "[Recital Match] BM25 top recitals: %s",
                ", ".join(r.metadata["id"] for r in results),
            )
        return results

    def _rerank_and_decorate(
        self,
        user_query: str,
        recital_docs: List[Document],
    ) -> List[Document]:
        logger.info("[Retriever] Reranking %d recital candidates", len(recital_docs))

        pairs = [[user_query, doc.page_content] for doc in recital_docs]
        ce_scores = self.cross_encoder.predict(pairs)

        top_indices = np.argsort(ce_scores)[::-1][:self.k]
        ranked_docs = [recital_docs[i] for i in top_indices]

        logger.info(
            "[Retriever] Final top-%d: %s",
            self.k,
            [f"{d.metadata.get('id')}({float(ce_scores[i]):.2f})"
             for d, i in zip(ranked_docs, top_indices)],
        )

        for doc in ranked_docs:
            header = _recital_header(doc.metadata["celex"], doc.page_content)
            body = _DISPLAY_NUM_RE.sub("", doc.page_content)
            doc.page_content = f"{header}\n{body}"

        return ranked_docs

    def _get_relevant_documents(
        self, user_query: str, *, run_manager: CallbackManagerForRetrieverRun = None  # noqa: ARG002
    ) -> List[Document]:
        classification = self.classifier.classify(user_query) if self.classifier else None
        target_acts = classification.acts if classification else []
        logger.info(
            "[Retriever] target_acts=%s intent=%s",
            target_acts,
            classification.intent if classification else None,
        )

        if not target_acts:
            logger.warning("[Retriever] No target acts classified — returning empty.")
            return []

        recital_docs = self._match_user_query_to_recitals(user_query, target_acts)
        if not recital_docs:
            return []

        return self._rerank_and_decorate(user_query, recital_docs)
