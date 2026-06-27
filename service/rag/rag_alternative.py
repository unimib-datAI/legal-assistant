import logging
import re
from collections import defaultdict
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
# Matches '\nid: <value>' in page_content produced by from_existing_graph
_PC_ID_RE = re.compile(r"\nid:\s*([^\n]+)")


def _recital_header(celex: str, recital_text: str) -> str:
    act_name = _CELEX_TO_ACT_NAME.get(celex, celex)
    m = _DISPLAY_NUM_RE.match(recital_text)
    if m:
        return f"[{act_name}, Recital {m.group(1)}]"
    return f"[{act_name}, Recital]"


def _doc_id(doc: Document) -> str:
    """Return the article id from metadata (sparse docs) or parsed from page_content (dense docs)."""
    mid = doc.metadata.get("id")
    if mid:
        return mid
    m = _PC_ID_RE.search(doc.page_content)
    return m.group(1).strip() if m else doc.page_content[:80]


class HybridRetriever(BaseRetriever):
    """Hybrid retriever: dense (vector) + sparse (BM25) article search fused with RRF.

    Recitals are retrieved via BM25 and judged by the same cross-encoder pass as the
    articles, then kept only if they clear `recital_score_threshold` — so irrelevant
    recitals are dropped instead of always padding the context.
    """

    graph: Any
    article_vector_store: Any
    classifier: Any = None
    top_k_dense: int = 10
    top_k_sparse: int = 10
    top_k_final: int = 5
    top_k_recitals: int = 3
    recital_score_threshold: float = 0.3
    rrf_k: int = 60
    use_reranker: bool = True
    cross_encoder: Any = CrossEncoder("BAAI/bge-reranker-v2-m3")
    _article_cache: Optional[dict] = None
    _recital_cache: Optional[dict] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _load_articles(self, acts: List[str]) -> dict:
        """Fetch all articles for the given acts and build a BM25 index + id lookup. Cached."""
        cache_key = tuple(sorted(acts))
        if self._article_cache is not None and cache_key in self._article_cache:
            return self._article_cache[cache_key]

        rows = self.graph.query(NodeQueries.GET_ARTICLES_BY_ACTS, params={"acts": acts})
        if not rows:
            data: dict = {"bm25": None, "by_id": {}}
        else:
            docs = [
                Document(
                    page_content=r["text"],
                    metadata={"id": r["id"], "title": r["title"], "act": r["act"], "type": "article"},
                )
                for r in rows
            ]
            bm25 = BM25Retriever.from_documents(docs)
            data = {"bm25": bm25, "by_id": {d.metadata["id"]: d for d in docs}}
            logger.info(
                "[Article Cache] Loaded %d articles + built BM25 index for acts %s",
                len(docs), acts,
            )

        if self._article_cache is None:
            self._article_cache = {}
        self._article_cache[cache_key] = data
        return data

    def _load_recitals(self, acts: List[str]) -> dict:
        """Fetch all recitals for the given acts and build a BM25 index. Cached."""
        cache_key = tuple(sorted(acts))
        if self._recital_cache is not None and cache_key in self._recital_cache:
            return self._recital_cache[cache_key]

        results = self.graph.query(NodeQueries.GET_RECITALS_BY_ACTS, params={"acts": acts})
        if not results:
            data = {"bm25": None}
        else:
            recital_docs = [
                Document(
                    page_content=r["text"],
                    metadata={"id": r["recital_id"], "celex": r["celex"], "type": "recital"},
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

    @staticmethod
    def _rrf_fusion(
        dense: List[Document],
        sparse: List[Document],
        k: int = 60,
    ) -> List[Document]:
        """Merge two ranked lists using Reciprocal Rank Fusion."""
        scores: dict = defaultdict(float)
        all_docs: dict = {}
        for rank, doc in enumerate(dense):
            key = _doc_id(doc)
            scores[key] += 1.0 / (k + rank + 1)
            all_docs[key] = doc
        for rank, doc in enumerate(sparse):
            key = _doc_id(doc)
            scores[key] += 1.0 / (k + rank + 1)
            # sparse docs carry full metadata (id, act, title) — prefer them for the same id
            all_docs[key] = doc
        return [all_docs[key] for key in sorted(scores, key=lambda x: -scores[x])]

    def _get_relevant_documents(
        self, user_query: str, *, run_manager: CallbackManagerForRetrieverRun = None  # noqa: ARG002
    ) -> List[Document]:
        classification = self.classifier.classify(user_query) if self.classifier else None
        target_acts = classification.acts if classification else []
        logger.info(
            "[HybridRetriever] target_acts=%s intent=%s",
            target_acts,
            classification.intent if classification else None,
        )

        if not target_acts:
            logger.warning("[HybridRetriever] No target acts classified — returning empty.")
            return []

        # Dense path: vector similarity search, then filter to target acts only
        raw_dense = self.article_vector_store.similarity_search(user_query, k=self.top_k_dense)
        dense_docs = [
            doc for doc in raw_dense
            if any(_doc_id(doc).startswith(act) for act in target_acts)
        ]
        logger.info("[HybridRetriever] Dense: %d article(s) (%d before act filter)", len(dense_docs), len(raw_dense))

        # Sparse path: BM25 over articles filtered by target acts
        article_data = self._load_articles(target_acts)
        bm25 = article_data.get("bm25")
        sparse_docs: List[Document] = []
        if bm25 is not None:
            bm25.k = self.top_k_sparse
            sparse_docs = bm25.invoke(user_query)
        logger.info("[HybridRetriever] Sparse: %d article(s)", len(sparse_docs))

        # RRF fusion → article candidates
        fused = self._rrf_fusion(dense_docs, sparse_docs, k=self.rrf_k)
        article_candidates = fused[: self.top_k_final * 2]

        # Ensure all article candidates have `id` in metadata; enrich from sparse corpus
        by_id = article_data.get("by_id", {})
        for doc in article_candidates:
            doc.metadata.setdefault("type", "article")
            if not doc.metadata.get("id"):
                doc.metadata["id"] = _doc_id(doc)
            if not doc.metadata.get("act") and doc.metadata["id"] in by_id:
                doc.metadata.update(by_id[doc.metadata["id"]].metadata)

        # Recital candidates: a wider BM25 pool to give the threshold room to choose
        recital_data = self._load_recitals(target_acts)
        bm25_r = recital_data.get("bm25")
        recital_candidates: List[Document] = []
        if bm25_r is not None:
            bm25_r.k = self.top_k_recitals * 2
            recital_candidates = bm25_r.invoke(user_query)

        candidates = article_candidates + recital_candidates
        if not candidates:
            return []

        # Single cross-encoder pass over articles + recitals → comparable scores
        if self.use_reranker:
            pairs = [[user_query, doc.page_content] for doc in candidates]
            ce_scores = self.cross_encoder.predict(pairs)
        else:
            # Preserve RRF/BM25 order: descending pseudo-scores by position
            ce_scores = np.arange(len(candidates), 0, -1, dtype=float)

        scored = list(zip(candidates, (float(s) for s in ce_scores)))
        article_scored = [(d, s) for d, s in scored if d.metadata.get("type") != "recital"]
        recital_scored = [(d, s) for d, s in scored if d.metadata.get("type") == "recital"]

        # Articles: guaranteed slots, top_k_final by score
        article_scored.sort(key=lambda x: -x[1])
        article_docs = [d for d, _ in article_scored[: self.top_k_final]]
        logger.info(
            "[HybridRetriever] Articles after reranking: %s",
            [f"{d.metadata.get('id')}({s:.2f})" for d, s in article_scored[: self.top_k_final]],
        )

        # Recitals: kept only if above threshold, capped at top_k_recitals
        recital_scored.sort(key=lambda x: -x[1])
        surviving = [(d, s) for d, s in recital_scored if s >= self.recital_score_threshold][: self.top_k_recitals]
        logger.info(
            "[HybridRetriever] Recitals kept %d/%d (threshold=%.2f): %s",
            len(surviving), len(recital_scored), self.recital_score_threshold,
            [f"{d.metadata.get('id')}({s:.2f})" for d, s in recital_scored],
        )

        # Decorate articles with act + title header
        for doc in article_docs:
            act_name = _CELEX_TO_ACT_NAME.get(
                doc.metadata.get("act", ""), doc.metadata.get("act", "")
            )
            title = doc.metadata.get("title", "")
            doc.page_content = f"[{act_name}, {title}]\n{doc.page_content}"

        # Decorate surviving recitals with their header
        recital_docs: List[Document] = []
        for doc, _ in surviving:
            header = _recital_header(doc.metadata["celex"], doc.page_content)
            body = _DISPLAY_NUM_RE.sub("", doc.page_content)
            doc.page_content = f"{header}\n{body}"
            recital_docs.append(doc)

        final_docs = article_docs + recital_docs
        logger.info(
            "[HybridRetriever] Final top-%d: %s",
            len(final_docs),
            [f"{d.metadata.get('id')}({d.metadata.get('type', '?')})" for d in final_docs],
        )
        return final_docs
