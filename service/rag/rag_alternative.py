import logging
import re
from collections import defaultdict
from typing import List, Any, Optional, Tuple

import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from sentence_transformers.cross_encoder import CrossEncoder

import config
from service.graph.query import NodeQueries
from service.rag.acts import CELEX_TO_ACT_NAME as _CELEX_TO_ACT_NAME
from service.rag.prompt import HYDE_PROMPT

logger = logging.getLogger(__name__)


class HyDEGenerator:
    """Generates act-grounded hypothetical legal passages for dense retrieval.

    With `iterations > 1` it samples multiple hypothetical documents (the LLM must
    use temperature > 0 for them to differ); the retriever averages their embeddings
    to reduce the variance of any single generated passage (Gao et al., 2022).
    """

    def __init__(self, llm: Any, iterations: int = 1):
        self.llm = llm
        self.iterations = iterations
        self._prompt = PromptTemplate.from_template(HYDE_PROMPT)

    def generate(self, query: str, acts_context: str) -> List[str]:
        text = self._prompt.format(query=query, acts=acts_context)
        return [self.llm.invoke(text).content.strip() for _ in range(self.iterations)]

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
    recitals are dropped instead of always padding the context. Set `use_recitals=False`
    to skip the recital branch entirely and return articles only.
    """

    graph: Any
    article_vector_store: Any
    classifier: Any = None
    hyde_generator: Any = None
    use_hyde: bool = True
    top_k_dense: int = 10
    top_k_sparse: int = 10
    top_k_final: int = 3
    top_k_recitals: int = 2
    recital_score_threshold: float = 0.3
    use_recitals: bool = True
    use_query_decomposition: bool = False
    max_sub_questions: int = 3
    rrf_k: int = 60
    use_reranker: bool = True
    cross_encoder: Any = CrossEncoder(
        config.RERANK_MODEL,
        device="cuda",
        trust_remote_code=config.RERANK_TRUST_REMOTE_CODE,
        model_kwargs={"dtype": config.RERANK_DTYPE},
    )
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
                    metadata={
                        "id": r["id"], "title": r["title"], "act": r["act"], "type": "article",
                        "chapter_number": r.get("chapter_number"),
                        "chapter_title": r.get("chapter_title"),
                    },
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
    def _rrf_fusion(*ranked_lists: List[Document], k: int = 60) -> List[Document]:
        """Merge N ranked lists using Reciprocal Rank Fusion.

        Lists are accumulated in order, so a later list wins the doc-object tie for the
        same id: callers pass sparse lists LAST because those docs carry full metadata
        (id, act, title) that dense docs lack.
        """
        scores: dict = defaultdict(float)
        all_docs: dict = {}
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked):
                key = _doc_id(doc)
                scores[key] += 1.0 / (k + rank + 1)
                all_docs[key] = doc
        return [all_docs[key] for key in sorted(scores, key=lambda x: -scores[x])]

    def _rerank_articles(
        self, user_query: str, article_candidates: List[Document]
    ) -> List[Tuple[Document, float]]:
        """Score each candidate article (whole text) against the query with the cross-encoder."""
        if not article_candidates:
            return []
        if not self.use_reranker:
            # Preserve RRF order: descending pseudo-scores by position.
            n = len(article_candidates)
            return [(doc, float(n - i)) for i, doc in enumerate(article_candidates)]

        pairs = [[user_query, doc.page_content] for doc in article_candidates]
        ce_scores = self.cross_encoder.predict(pairs)
        return [(doc, float(s)) for doc, s in zip(article_candidates, ce_scores)]

    def _rerank_recitals(
        self, user_query: str, recital_candidates: List[Document]
    ) -> List[Tuple[Document, float]]:
        """Score recitals whole against the query (short, single-paragraph — no dilution)."""
        if not recital_candidates:
            return []
        if not self.use_reranker:
            n = len(recital_candidates)
            return [(doc, float(n - i)) for i, doc in enumerate(recital_candidates)]
        pairs = [[user_query, doc.page_content] for doc in recital_candidates]
        ce_scores = self.cross_encoder.predict(pairs)
        return [(doc, float(score)) for doc, score in zip(recital_candidates, ce_scores)]

    def _dense_search(self, query_text: str, target_acts: List[str]) -> List[Document]:
        """Dense article search for one query, HyDE-expanded when enabled, act-filtered.

        HyDE passages are averaged into a single mean vector *within* this query; fusion
        across queries happens at the ranked-list level (RRF), not by averaging vectors.
        """
        if self.use_hyde and self.hyde_generator is not None:
            acts_context = ", ".join(_CELEX_TO_ACT_NAME.get(a, a) for a in target_acts)
            hyde_docs = self.hyde_generator.generate(query_text, acts_context)
            doc_vectors = self.article_vector_store.embedding.embed_documents(hyde_docs)
            mean_vector = np.mean(np.asarray(doc_vectors), axis=0).tolist()
            raw_dense = self.article_vector_store.similarity_search_by_vector(
                mean_vector, k=self.top_k_dense, query=query_text
            )
        else:
            raw_dense = self.article_vector_store.similarity_search(query_text, k=self.top_k_dense)
        return [
            doc for doc in raw_dense
            if any(_doc_id(doc).startswith(act) for act in target_acts)
        ]

    def _get_relevant_documents(
        self, user_query: str, *, run_manager: CallbackManagerForRetrieverRun = None  # noqa: ARG002
    ) -> List[Document]:
        classification = self.classifier.classify(user_query) if self.classifier else None
        target_acts = classification.acts if classification else []
        logger.info(
            "[HybridRetriever] target_acts=%s intent=%s act_scores=%s",
            target_acts,
            classification.intent if classification else None,
            classification.act_scores if classification else None,
        )

        if not target_acts:
            logger.warning("[HybridRetriever] No target acts classified — returning empty.")
            return []

        # Search queries: the original question, plus decomposed sub-questions when
        # decomposition is enabled and the classifier produced any. Each is retrieved
        # independently (dense HyDE + sparse BM25); all lists are RRF-fused so every
        # facet's provisions get a chance in the candidate pool.
        search_queries = [user_query]
        if self.use_query_decomposition and classification and classification.sub_questions:
            seen = {user_query.strip().lower()}
            for sq in classification.sub_questions[: self.max_sub_questions]:
                key = sq.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    search_queries.append(sq)
            logger.info(
                "[HybridRetriever] Decomposition: %d sub-question(s): %s",
                len(search_queries) - 1, search_queries[1:],
            )

        article_data = self._load_articles(target_acts)
        bm25 = article_data.get("bm25")
        if bm25 is not None:
            bm25.k = self.top_k_sparse

        dense_lists: List[List[Document]] = []
        sparse_lists: List[List[Document]] = []
        for q in search_queries:
            dense_docs = self._dense_search(q, target_acts)
            dense_lists.append(dense_docs)
            sparse_docs = bm25.invoke(q) if bm25 is not None else []
            sparse_lists.append(sparse_docs)
            logger.info(
                "[HybridRetriever] Query %r → dense %d, sparse %d",
                q[:60], len(dense_docs), len(sparse_docs),
            )

        # Sparse lists LAST so their fuller metadata wins the RRF doc-object tie.
        fused = self._rrf_fusion(*dense_lists, *sparse_lists, k=self.rrf_k)
        # Widen the pre-rerank pool with the number of search queries so sub-question
        # provisions are not cut before the cross-encoder scores them.
        article_candidates = fused[: self.top_k_final * 2 * len(search_queries)]

        # Ensure all article candidates have `id` in metadata; enrich from sparse corpus
        by_id = article_data.get("by_id", {})
        for doc in article_candidates:
            doc.metadata.setdefault("type", "article")
            if not doc.metadata.get("id"):
                doc.metadata["id"] = _doc_id(doc)
            if not doc.metadata.get("act") and doc.metadata["id"] in by_id:
                doc.metadata.update(by_id[doc.metadata["id"]].metadata)

        # Recital candidates: a wider BM25 pool to give the threshold room to choose.
        # Skipped entirely when recital search is disabled.
        recital_candidates: List[Document] = []
        if self.use_recitals:
            recital_data = self._load_recitals(target_acts)
            bm25_r = recital_data.get("bm25")
            if bm25_r is not None:
                bm25_r.k = self.top_k_recitals * 2
                recital_candidates = bm25_r.invoke(user_query)

        if not article_candidates and not recital_candidates:
            return []

        # Articles and recitals are reranked (whole text) and ranked/selected independently —
        # recitals never competed with articles for the guaranteed article slots.
        article_scored = self._rerank_articles(user_query, article_candidates)
        recital_scored = self._rerank_recitals(user_query, recital_candidates)

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

        # Decorate articles with act + chapter + title header. The chapter lets the
        # synthesis LLM resolve questions scoped to a specific Chapter (e.g. "the
        # personal scope of Chapter II") to the right provision.
        for doc in article_docs:
            act_name = _CELEX_TO_ACT_NAME.get(
                doc.metadata.get("act", ""), doc.metadata.get("act", "")
            )
            title = doc.metadata.get("title", "")
            chapter = doc.metadata.get("chapter_number")
            if chapter:
                chapter_title = doc.metadata.get("chapter_title") or ""
                header = f"[{act_name}, Chapter {chapter} — {chapter_title}, {title}]"
            else:
                header = f"[{act_name}, {title}]"
            doc.page_content = f"{header}\n{doc.page_content}"

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
