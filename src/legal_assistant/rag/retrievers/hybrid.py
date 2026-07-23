import logging
from collections import Counter, defaultdict
from typing import List, Any, Optional, Tuple

import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from sentence_transformers.cross_encoder import CrossEncoder

from legal_assistant import config
from legal_assistant.graph.queries import CaseLawQueries, NodeQueries
from legal_assistant.rag.acts import CELEX_TO_ACT_NAME, celex_instrument_and_numbers
from legal_assistant.rag.citations import cited_articles
from legal_assistant.rag.documents import (
    copy_doc,
    decorate_annex,
    decorate_article,
    decorate_case_law,
    decorate_obligation,
    decorate_recital,
    doc_id,
    neighbour_ids,
)

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """Hybrid retriever: dense (vector) + sparse (BM25) article search fused with RRF.

    Recitals are retrieved via BM25 and judged by the same cross-encoder pass as the
    articles, then kept only if they clear `recital_score_threshold`, so irrelevant
    recitals are dropped instead of always padding the context. Set `use_recitals=False`
    to skip the recital branch entirely and return articles only.

    On an INTERPRETIVE query (see `intent_classifier`) a second branch searches CJEU
    judgment paragraphs, restricted to judgments that INTERPRETS one of the target acts.
    It feeds the answer twice:

    * the surviving paragraphs are appended to the context in their own slots, and
    * the provisions those paragraphs *cite in their own text* are fused back into the
      article RRF as an extra ranked list, a *graph boost* on the act branch.

    The act branch is never made to depend on the case law branch: articles keep their
    `top_k_final` guaranteed slots, and a query that retrieves no case law behaves exactly
    as it does today. There is therefore no "no match" fallback path.
    """

    graph: Any
    article_vector_store: Any
    case_law_vector_store: Any = None
    classifier: Any = None
    hyde_generator: Any = None
    use_hyde: bool = True
    top_k_dense: int = 10
    top_k_sparse: int = 10
    top_k_final: int = 3
    top_k_recitals: int = 2
    recital_score_threshold: float = 0.3
    use_recitals: bool = True
    # The obligations branch, like the annex branch, is off by default and carries no
    # ParamSpec. It activates only when the classifier names an addressee role, which the
    # active v5 classification prompt does not, so it is inert until a prompt that scores
    # addressees is wired in. See docs/superpowers for the activation plan.
    use_obligations: bool = False
    top_k_obligations: int = 5
    obligation_score_threshold: float = 0.3
    # The annex branch is deliberately unreachable from the application: it carries no
    # ParamSpec, so `pipeline.py` neither defaults it nor accepts it as an override. The
    # obligations work needs Annex *nodes*, not annex retrieval, because an obligation
    # extracted from an annex carries its own text and cites itself through `point_label`.
    # Give it a ParamSpec, or read it from `config`, to turn it on.
    use_annexes: bool = False
    top_k_annexes: int = 2
    annex_score_threshold: float = 0.3
    use_case_law: bool = True
    top_k_case_law: int = 5
    case_law_score_threshold: float = 0.3
    case_law_neighbours: int = 2
    guarantee_operative: bool = True
    top_k_bridge: int = 3
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
    _annex_cache: Optional[dict] = None
    _case_law_cache: Optional[dict] = None

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

    def _load_annex_points(self, acts: List[str]) -> dict:
        """Fetch annex points for the given acts and build a BM25 index. Cached.

        Only the AI Act has annexes, so for the other three this loads nothing and the branch
        is empty rather than absent: there is no fallback path to take.
        """
        cache_key = tuple(sorted(acts))
        if self._annex_cache is not None and cache_key in self._annex_cache:
            return self._annex_cache[cache_key]

        rows = self.graph.query(NodeQueries.GET_ANNEX_POINTS_BY_ACTS, params={"acts": acts})
        if not rows:
            data: dict = {"bm25": None}
        else:
            docs = [
                Document(
                    page_content=r["text"],
                    metadata={
                        "id": r["id"], "celex": r["celex"], "type": "annex",
                        "point_label": r.get("point_label"),
                        "section_heading": r.get("section_heading"),
                        "annex_number": r.get("annex_number"),
                        "annex_title": r.get("annex_title"),
                    },
                )
                for r in rows
            ]
            data = {"bm25": BM25Retriever.from_documents(docs)}
            logger.info(
                "[Annex Cache] Loaded %d annex point(s) + built BM25 index for acts %s",
                len(docs), acts,
            )

        if self._annex_cache is None:
            self._annex_cache = {}
        self._annex_cache[cache_key] = data
        return data

    def _select_annexes(self, user_query: str, acts: List[str]) -> List[Tuple[Document, float]]:
        """The annex branch end to end: search, rank, threshold, cap.

        Sparse only, like the recital branch: an annex point is a short, densely referential
        span ("the AI systems listed in point 1(a)"), which BM25 handles and a hypothetical
        passage written in the register of an article does not.
        """
        data = self._load_annex_points(acts)
        bm25 = data.get("bm25")
        if bm25 is None:
            return []

        bm25.k = self.top_k_annexes * 2
        candidates = [copy_doc(d) for d in bm25.invoke(user_query)]
        if not candidates:
            return []

        scored = self._rerank_annexes(user_query, candidates)
        scored.sort(key=lambda pair: -pair[1])
        kept = [
            (doc, score) for doc, score in scored if score >= self.annex_score_threshold
        ][: self.top_k_annexes]

        logger.info(
            "[HybridRetriever] Annex points kept %d/%d (threshold=%.2f): %s",
            len(kept), len(scored), self.annex_score_threshold,
            [f"{doc_id(d)}({s:.2f})" for d, s in scored[:8]],
        )
        return kept

    def _rerank_annexes(
        self, user_query: str, candidates: List[Document]
    ) -> List[Tuple[Document, float]]:
        """Score annex points whole against the query (short spans, no dilution)."""
        if not candidates:
            return []
        if not self.use_reranker:
            n = len(candidates)
            return [(doc, float(n - i)) for i, doc in enumerate(candidates)]
        pairs = [[user_query, doc.page_content] for doc in candidates]
        ce_scores = self.cross_encoder.predict(pairs)
        return [(doc, float(score)) for doc, score in zip(candidates, ce_scores)]

    def _select_obligations(
        self, user_query: str, acts: List[str], addressees: List[str]
    ) -> List[Tuple[Document, float]]:
        """The obligations branch: filter by role on the graph, rerank, threshold, cap.

        Filtered on the graph rather than searched: the ``IS_A`` walk narrows the set to the
        classified role and its qualified forms, which is small enough to rerank in full. No
        addressee, no branch: it degrades to nothing rather than guessing who the query is
        about.
        """
        if not addressees:
            return []

        rows = self.graph.query(
            NodeQueries.GET_OBLIGATIONS_FOR_ACTORS,
            params={"acts": acts, "actors": addressees},
        )
        if not rows:
            return []

        candidates = [
            Document(
                page_content=self._render_obligation(row),
                metadata={
                    "id": row["id"], "source_id": row["source_id"], "actor": row["actor"],
                    "celex": row["source_id"].split("_", 1)[0].split("anx_")[0],
                    "weakest_method": row["weakest_method"], "type": "obligation",
                },
            )
            for row in rows
        ]

        scored = self._rerank_obligations(user_query, candidates)
        scored.sort(key=lambda pair: -pair[1])
        kept = [
            (doc, score) for doc, score in scored
            if score >= self.obligation_score_threshold
        ][: self.top_k_obligations]

        logger.info(
            "[HybridRetriever] Obligations kept %d/%d for %s (threshold=%.2f)",
            len(kept), len(scored), addressees, self.obligation_score_threshold,
        )
        return kept

    @staticmethod
    def _render_obligation(row: dict) -> str:
        parts = [row["predicate_text"] or ""]
        if row.get("target"):
            parts.append(f"({row['target']})")
        if row.get("specification"):
            parts.append(f"[{row['specification']}]")
        return " ".join(p for p in parts if p)

    def _rerank_obligations(
        self, user_query: str, candidates: List[Document]
    ) -> List[Tuple[Document, float]]:
        """Score each obligation (its rendered text) against the query."""
        if not candidates:
            return []
        if not self.use_reranker:
            n = len(candidates)
            return [(doc, float(n - i)) for i, doc in enumerate(candidates)]
        pairs = [[user_query, doc.page_content] for doc in candidates]
        ce_scores = self.cross_encoder.predict(pairs)
        return [(doc, float(score)) for doc, score in zip(candidates, ce_scores)]

    def _load_case_law_paragraphs(self, acts: List[str]) -> dict:
        """Fetch judgment paragraphs for the given acts and build a BM25 index + id lookup. Cached."""
        cache_key = tuple(sorted(acts))
        if self._case_law_cache is not None and cache_key in self._case_law_cache:
            return self._case_law_cache[cache_key]

        rows = self.graph.query(CaseLawQueries.GET_CASE_LAW_PARAGRAPHS_BY_ACTS, params={"acts": acts})
        if not rows:
            data: dict = {"bm25": None, "by_id": {}}
        else:
            docs = [
                Document(
                    page_content=r["text"],
                    metadata={
                        "id": r["id"], "celex": r["celex"], "case_number": r["case_number"],
                        "number": r["number"], "is_operative": r["is_operative"],
                        "section_heading": r["section_heading"], "type": "case_law",
                    },
                )
                for r in rows
            ]
            data = {
                "bm25": BM25Retriever.from_documents(docs),
                "by_id": {d.metadata["id"]: d for d in docs},
            }
            logger.info(
                "[Case Law Cache] Loaded %d judgment paragraphs + built BM25 index for acts %s",
                len(docs), acts,
            )

        if self._case_law_cache is None:
            self._case_law_cache = {}
        self._case_law_cache[cache_key] = data
        return data

    def _case_law_search(self, user_query: str, data: dict) -> List[Document]:
        """Dense + BM25 search over judgment paragraphs, RRF-fused.

        No HyDE here: the hypothetical passage is generated in the register of the
        legislation, which is not how a judgment reads.
        """
        by_id = data.get("by_id", {})
        dense: List[Document] = []
        if self.case_law_vector_store is not None:
            raw = self.case_law_vector_store.similarity_search(user_query, k=self.top_k_dense)
            # Dense hits carry no usable metadata; re-attach it from the sparse corpus and
            # drop anything outside the act-scoped candidate set.
            dense = [by_id[doc_id(d)] for d in raw if doc_id(d) in by_id]

        bm25 = data.get("bm25")
        sparse: List[Document] = []
        if bm25 is not None:
            bm25.k = self.top_k_sparse
            sparse = bm25.invoke(user_query)

        fused = self._rrf_fusion(dense, sparse, k=self.rrf_k)
        logger.info(
            "[HybridRetriever] Case law: dense %d, sparse %d → %d candidates",
            len(dense), len(sparse), len(fused),
        )
        return [copy_doc(d) for d in fused[: self.top_k_case_law * 3]]

    def _expand_neighbours(
        self, seeds: List[Document], by_id: dict, already_scored: set
    ) -> List[Document]:
        """Paragraphs reading around the seeds, not already in the candidate pool.

        Search matches a judgment's *topic*, which is argued at length in the run-up, and
        misses its *holding*, which is a terse sentence with almost no lexical surface
        ("It follows that … is therefore invalid.") sitting immediately after. Reading a
        short window around each hit recovers the conclusion the argument was building to.
        """
        found: List[Document] = []
        seen = set(already_scored)
        for doc in seeds:
            for neighbour_id in neighbour_ids(doc_id(doc), self.case_law_neighbours):
                if neighbour_id in by_id and neighbour_id not in seen:
                    seen.add(neighbour_id)
                    found.append(copy_doc(by_id[neighbour_id]))
        return found

    def _ensure_operative(
        self, user_query: str, kept: List[Tuple[Document, float]], by_id: dict
    ) -> List[Tuple[Document, float]]:
        """Guarantee the operative part of the top judgment a slot, if it earns one.

        The operative part *is* the holding, the paragraphs the Court hands down as its
        answer, so on a question about what a judgment decided it is the highest-value
        passage by construction. It is fetched rather than searched because it can lose the
        candidate stage outright: it opens with a formal recitation of the instrument under
        review, which matches a question about the ruling poorly.

        Additive: this is a slot beyond `top_k_case_law`, and it is skipped when the kept set
        already holds an operative passage or when none clears the threshold.
        """
        if not kept or any(doc.metadata.get("is_operative") for doc, _ in kept):
            return kept

        top_celex = kept[0][0].metadata.get("celex")
        kept_ids = {doc_id(doc) for doc, _ in kept}
        operative = [
            copy_doc(doc) for doc in by_id.values()
            if doc.metadata.get("is_operative")
            and doc.metadata.get("celex") == top_celex
            and doc.metadata["id"] not in kept_ids
        ]
        if not operative:
            return kept

        best_doc, best_score = max(
            self._rerank_case_law(user_query, operative), key=lambda pair: pair[1]
        )
        if best_score < self.case_law_score_threshold:
            return kept
        logger.info(
            "[HybridRetriever] Operative slot: %s(%.2f)", doc_id(best_doc), best_score,
        )
        return kept + [(best_doc, best_score)]

    def _select_case_law(
        self, user_query: str, acts: List[str]
    ) -> List[Tuple[Document, float]]:
        """The case law branch end to end: search, read around the hits, rank, threshold, cap."""
        data = self._load_case_law_paragraphs(acts)
        by_id = data.get("by_id", {})
        if not by_id:
            return []

        candidates = self._case_law_search(user_query, data)
        if not candidates:
            return []

        scored = self._rerank_case_law(user_query, candidates)
        scored.sort(key=lambda pair: -pair[1])

        # Expand around the hits that would survive on their own, then let the reranker judge
        # the neighbours on the same footing: a neighbour is kept only if it outscores what
        # it displaces, so expansion cannot dilute a good result.
        seeds = [
            doc for doc, score in scored[: self.top_k_case_law]
            if score >= self.case_law_score_threshold
        ]
        neighbours = self._expand_neighbours(seeds, by_id, {doc_id(d) for d, _ in scored})
        if neighbours:
            scored.extend(self._rerank_case_law(user_query, neighbours))
            scored.sort(key=lambda pair: -pair[1])

        kept = [
            (doc, score) for doc, score in scored
            if score >= self.case_law_score_threshold
        ][: self.top_k_case_law]

        if self.guarantee_operative:
            kept = self._ensure_operative(user_query, kept, by_id)

        logger.info(
            "[HybridRetriever] Case law kept %d/%d (+%d neighbour(s), threshold=%.2f): %s",
            len(kept), len(scored), len(neighbours), self.case_law_score_threshold,
            [f"{doc_id(d)}({s:.2f})" for d, s in scored[:8]],
        )
        return kept

    def _bridge_articles(
        self, kept_case_law: List[Tuple[Document, float]], acts: List[str], by_id: dict
    ) -> List[Document]:
        """Articles the retrieved judgment passages cite in their own text.

        Ranked by the rerank score of the passage that cites them, never by how often an
        article is cited. Frequency is what the INTERPRETS bridge used and it simply
        resurfaced the corpus's most-litigated provisions; here the article inherits the
        relevance of the passage that called for it.
        """
        if not kept_case_law or not by_id:
            return []

        # "of that regulation" is only unambiguous while one act of its instrument type is in
        # play. With a Regulation and a Directive targeted it still resolves; with two
        # Regulations it would resolve against both, so it is switched off per act.
        instruments = Counter(
            parsed[0] for parsed in (celex_instrument_and_numbers(a) for a in acts)
            if parsed is not None
        )

        best: dict = {}
        for doc, score in kept_case_law:
            for celex in acts:
                parsed = celex_instrument_and_numbers(celex)
                resolve_anaphora = parsed is not None and instruments[parsed[0]] == 1
                for article_id in cited_articles(
                    doc.page_content, celex, resolve_anaphora=resolve_anaphora
                ):
                    if article_id in by_id and score > best.get(article_id, float("-inf")):
                        best[article_id] = score

        ranked = sorted(best, key=lambda article_id: -best[article_id])[: self.top_k_bridge]
        logger.info(
            "[HybridRetriever] Bridge: %d passage(s) cite %d article(s) → %s",
            len(kept_case_law), len(best),
            [f"{article_id}({best[article_id]:.2f})" for article_id in ranked],
        )
        return [copy_doc(by_id[article_id]) for article_id in ranked]

    def _rerank_case_law(
        self, user_query: str, candidates: List[Document]
    ) -> List[Tuple[Document, float]]:
        """Score judgment paragraphs against the query (single paragraphs, no dilution)."""
        if not candidates:
            return []
        if not self.use_reranker:
            n = len(candidates)
            return [(doc, float(n - i)) for i, doc in enumerate(candidates)]
        pairs = [[user_query, doc.page_content] for doc in candidates]
        ce_scores = self.cross_encoder.predict(pairs)
        return [(doc, float(score)) for doc, score in zip(candidates, ce_scores)]

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
                key = doc_id(doc)
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
        """Score recitals whole against the query (short, single-paragraph, no dilution)."""
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
            acts_context = ", ".join(CELEX_TO_ACT_NAME.get(a, a) for a in target_acts)
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
            if any(doc_id(doc).startswith(act) for act in target_acts)
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
            logger.warning("[HybridRetriever] No target acts classified, returning empty.")
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

        # Case law branch: only on INTERPRETIVE queries, and always alongside the act branch
        # rather than in place of it. It runs to completion first (including its rerank)
        # because the bridge ranks each cited article by the score of the passage citing it,
        # and that score does not exist until the case law has been judged.
        is_interpretive = classification is not None and classification.intent == "INTERPRETIVE"
        kept_case_law: List[Tuple[Document, float]] = []
        bridge_list: List[Document] = []
        if is_interpretive and self.use_case_law:
            kept_case_law = self._select_case_law(user_query, target_acts)
            bridge_list = self._bridge_articles(
                kept_case_law, target_acts, article_data.get("by_id", {})
            )

        dense_lists: List[List[Document]] = []
        sparse_lists: List[List[Document]] = []
        for q in search_queries:
            dense_docs = self._dense_search(q, target_acts)
            dense_lists.append(dense_docs)
            sparse_docs = [copy_doc(d) for d in bm25.invoke(q)] if bm25 is not None else []
            sparse_lists.append(sparse_docs)
            logger.info(
                "[HybridRetriever] Query %r → dense %d, sparse %d",
                q[:60], len(dense_docs), len(sparse_docs),
            )

        # Sparse lists LAST so their fuller metadata wins the RRF doc-object tie. The bridge is
        # just one more ranked list: when it is empty (no case law, or none retrieved) the
        # fusion is bit-for-bit what it was before the case law branch existed.
        fused = self._rrf_fusion(*dense_lists, bridge_list, *sparse_lists, k=self.rrf_k)
        # Widen the pre-rerank pool with the number of search queries so sub-question
        # provisions are not cut before the cross-encoder scores them.
        article_candidates = fused[: self.top_k_final * 2 * len(search_queries)]

        # Ensure all article candidates have `id` in metadata; enrich from sparse corpus
        by_id = article_data.get("by_id", {})
        for doc in article_candidates:
            doc.metadata.setdefault("type", "article")
            if not doc.metadata.get("id"):
                doc.metadata["id"] = doc_id(doc)
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
                recital_candidates = [copy_doc(d) for d in bm25_r.invoke(user_query)]

        kept_annexes = self._select_annexes(user_query, target_acts) if self.use_annexes else []

        kept_obligations: List[Tuple[Document, float]] = []
        if self.use_obligations:
            addressees = classification.addressees if classification else []
            # When the shared classifier did not score addressees (its prompt does not), the
            # branch does its own role classification rather than guessing.
            if not addressees and self.classifier is not None:
                addressees, _ = self.classifier.classify_addressees(user_query)
            kept_obligations = self._select_obligations(user_query, target_acts, addressees)

        if (not article_candidates and not recital_candidates
                and not kept_case_law and not kept_annexes and not kept_obligations):
            return []

        # Articles and recitals are reranked (whole text) and selected independently: neither
        # recitals nor judgments ever compete with articles for the guaranteed article slots.
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

        article_docs = [decorate_article(doc) for doc in article_docs]
        recital_docs = [decorate_recital(doc) for doc, _ in surviving]
        annex_docs = [decorate_annex(doc) for doc, _ in kept_annexes]
        obligation_docs = [decorate_obligation(doc) for doc, _ in kept_obligations]
        case_law_docs = [decorate_case_law(doc) for doc, _ in kept_case_law]

        final_docs = article_docs + recital_docs + annex_docs + obligation_docs + case_law_docs
        logger.info(
            "[HybridRetriever] Final top-%d: %s",
            len(final_docs),
            [f"{d.metadata.get('id')}({d.metadata.get('type', '?')})" for d in final_docs],
        )
        return final_docs
