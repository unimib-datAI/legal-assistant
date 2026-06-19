import logging
import re
from collections import Counter
from typing import List, Any, Optional

import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

from service.graph.query import NodeQueries

logger = logging.getLogger(__name__)

_CELEX_TO_ACT_NAME = {
    "32022R0868": "Data Governance Act",
    "32023R2854": "Data Act",
    "32024R1689": "AI Act",
    "32016R0679": "GDPR",
}

_ARTICLE_ID_RE = re.compile(r"^(.+?)art_(\d+)$", re.IGNORECASE)
_DISPLAY_NUM_RE = re.compile(r"^\((\d+)\)\s*")

_ROMAN_NUMERALS = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def _int_to_roman(n: int) -> str:
    result = []
    for value, numeral in _ROMAN_NUMERALS:
        while n >= value:
            result.append(numeral)
            n -= value
    return "".join(result)


def _source_header(article_id: str, article_title: str) -> str:
    """Build a '[Act, Article N — Title]' prefix from the article node ID."""
    m = _ARTICLE_ID_RE.match(article_id)
    if not m:
        return f"[{article_title}]"
    celex, art_num = m.group(1), m.group(2)
    act_name = _CELEX_TO_ACT_NAME.get(celex, celex)
    return f"[{act_name}, Article {int(art_num)} — {article_title}]"


def _recital_header(celex: str, recital_text: str) -> str:
    """Build a '[Act, Recital N]' prefix using the display label parsed from the recital text."""
    act_name = _CELEX_TO_ACT_NAME.get(celex, celex)
    m = _DISPLAY_NUM_RE.match(recital_text)
    if m:
        return f"[{act_name}, Recital {m.group(1)}]"
    return f"[{act_name}, Recital]"


class ArticleTraversalRetriever(BaseRetriever):
    """Retriever that matches the query against article titles, then traverses
    the graph to collect all paragraphs of the top-matched articles."""

    graph: Any
    k: int = 10
    top_k_articles: int = 10
    top_k_recitals: int = 2
    article_similarity_threshold: float = 0.2
    article_score_boost: float = 10.0
    recital_score_boost: float = 3.0
    embedding_model: Any = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder: Any = CrossEncoder("BAAI/bge-reranker-v2-m3")
    classifier: Any = None
    _article_cache: Optional[dict] = None
    _recital_cache: Optional[dict] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _load_article_titles(self, acts: List[str]) -> dict:
        """Fetch and embed article titles for the given acts. Results are cached."""
        cache_key = tuple(sorted(acts))

        if self._article_cache is not None and cache_key in self._article_cache:
            return self._article_cache[cache_key]

        results = self.graph.query(
            NodeQueries.GET_ARTICLE_TITLES_BY_ACTS,
            params={"acts": acts}
        )

        if not results:
            data = {"ids": [], "titles": [], "embeddings": np.array([])}
        else:
            article_ids = [r["article_id"] for r in results]
            titles = [r["article_title"] for r in results]
            embeddings = self.embedding_model.encode(titles, show_progress_bar=False)
            data = {"ids": article_ids, "titles": titles, "embeddings": np.array(embeddings)}
            logger.info("[Article Cache] Loaded %d article titles for acts %s", len(titles), acts)

        if self._article_cache is None:
            self._article_cache = {}
        self._article_cache[cache_key] = data
        return data

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
            logger.info("[Recital Cache] Loaded %d recitals + built BM25 index for acts %s", len(recital_docs), acts)

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

    def _match_user_query_to_article_titles(
        self, user_query: str, target_acts: List[str]
    ) -> List[tuple[str, float]]:
        """Return (article_id, similarity_score) for the top-k most similar articles."""
        article_data = self._load_article_titles(target_acts)

        if not article_data["ids"]:
            logger.info("[Article Match] No articles found for acts %s", target_acts)
            return []

        query_embedding = self.embedding_model.encode(user_query, show_progress_bar=False)
        similarities = cosine_similarity([query_embedding], article_data["embeddings"])[0]

        ranked_indices = np.argsort(similarities)[::-1]
        matched = [
            (article_data["ids"][i], float(similarities[i]))
            for i in ranked_indices
            if similarities[i] >= self.article_similarity_threshold
        ][:self.top_k_articles]

        logger.info(
            "[Article Match] Top articles: %s",
            ", ".join(f"{article_id}({score:.3f})" for article_id, score in matched),
        )

        return matched

    _STRUCTURAL_FALLBACK_SCORE = 0.5
    _SPECIFIC_QUESTION_TOP_K = 8
    _SCOPE_OF_ACT_MAX_ARTICLE_NUM = 3
    _SCOPE_OF_CHAPTER_INTRO_ARTICLE_NUM = 1
    _DEFINITION_LOOKUP_MIN_MATCH_SCORE = 0.3
    _DEFINITION_LOOKUP_FALLBACK_TOP_K_RECITALS = 4

    def _paragraphs_to_docs(self, results: List[dict]) -> List[Document]:
        return [
            Document(
                page_content=r["text"],
                metadata={
                    "id": r["id"],
                    "article_id": r["article_id"],
                    "article_title": r["article_title"],
                    "source": "article_traversal",
                },
            )
            for r in results
        ]

    def _fetch_by_semantic_match(
        self, user_query: str, target_acts: List[str], top_k_articles: int
    ) -> tuple[List[Document], dict]:
        """Semantic match on article summaries, then traverse to paragraphs.
        Used by ENUMERATION (default) and SPECIFIC_QUESTION (narrower top_k)."""
        original_top_k = self.top_k_articles
        self.top_k_articles = top_k_articles
        try:
            matched = self._match_user_query_to_article_titles(user_query, target_acts)
        finally:
            self.top_k_articles = original_top_k

        if not matched:
            return [], {}

        article_scores = {art_id: score for art_id, score in matched}
        results = self.graph.query(
            NodeQueries.GET_PARAGRAPHS_BY_ARTICLES,
            params={"article_ids": list(article_scores.keys())},
        )
        docs = self._paragraphs_to_docs(results)
        self._log_paragraph_distribution("Semantic Match", results)
        return docs, article_scores

    def _fetch_scope_of_act(self, target_acts: List[str]) -> tuple[List[Document], dict]:
        """Fetch the general-provisions articles (typically Art. 1-3) of each target act."""
        results = self.graph.query(
            NodeQueries.GET_GENERAL_PROVISIONS_BY_ACTS,
            params={"acts": target_acts, "max_article_num": self._SCOPE_OF_ACT_MAX_ARTICLE_NUM},
        )
        docs = self._paragraphs_to_docs(results)
        article_scores = {r["article_id"]: self._STRUCTURAL_FALLBACK_SCORE for r in results}
        self._log_paragraph_distribution("Scope of Act", results)
        return docs, article_scores

    def _fetch_scope_of_chapter(
        self, target_acts: List[str], chapter_number: int
    ) -> tuple[List[Document], dict]:
        """Fetch all paragraphs of articles in the target Chapter, plus Article 1
        (Subject matter & scope). Article 2 (Definitions) is intentionally excluded
        because its broad term coverage tends to pull the cross-encoder toward
        definitions belonging to *other* chapters' subject matter."""
        chapter_roman = _int_to_roman(chapter_number)
        chapter_results = self.graph.query(
            NodeQueries.GET_CHAPTER_PARAGRAPHS,
            params={"acts": target_acts, "chapter_number": chapter_roman},
        )
        intro_results = self.graph.query(
            NodeQueries.GET_GENERAL_PROVISIONS_BY_ACTS,
            params={"acts": target_acts, "max_article_num": self._SCOPE_OF_CHAPTER_INTRO_ARTICLE_NUM},
        )

        seen = set()
        merged = []
        for r in chapter_results + intro_results:
            if r["id"] not in seen:
                seen.add(r["id"])
                merged.append(r)

        docs = self._paragraphs_to_docs(merged)
        article_scores = {r["article_id"]: self._STRUCTURAL_FALLBACK_SCORE for r in merged}
        self._log_paragraph_distribution(
            f"Scope of Chapter {chapter_number} ({chapter_roman})", merged
        )
        return docs, article_scores

    def _fetch_definition_lookup(
        self, user_query: str, target_acts: List[str]
    ) -> tuple[List[Document], dict]:
        """Semantic match on article summaries. If a dedicated article emerges with
        sufficient similarity, also add the act's Definitions article. Otherwise
        (weak match), skip Definitions — they would flood the pool with unrelated
        terms — and let the recital BM25 match (boosted) carry the interpretive load."""
        docs, article_scores = self._fetch_by_semantic_match(
            user_query, target_acts, top_k_articles=self.top_k_articles
        )

        top_match_score = max(article_scores.values(), default=0.0)
        if top_match_score < self._DEFINITION_LOOKUP_MIN_MATCH_SCORE:
            logger.info(
                "[Definition Lookup] top match score %.3f < %.2f — skipping Definitions, "
                "relying on semantic match + recitals (no dedicated article emerged).",
                top_match_score, self._DEFINITION_LOOKUP_MIN_MATCH_SCORE,
            )
            self._log_paragraph_distribution("Definition Lookup (no defs)", [
                {"article_id": doc.metadata["article_id"]} for doc in docs
            ])
            return docs, article_scores

        definitions_results = self.graph.query(
            NodeQueries.GET_DEFINITIONS_BY_ACTS,
            params={"acts": target_acts},
        )
        seen = {doc.metadata["id"] for doc in docs}
        for r in definitions_results:
            if r["id"] not in seen:
                docs.append(Document(
                    page_content=r["text"],
                    metadata={
                        "id": r["id"],
                        "article_id": r["article_id"],
                        "article_title": r["article_title"],
                        "source": "article_traversal",
                    },
                ))
                article_scores.setdefault(r["article_id"], self._STRUCTURAL_FALLBACK_SCORE)
        self._log_paragraph_distribution("Definition Lookup (incl. defs)", [
            {"article_id": doc.metadata["article_id"]} for doc in docs
        ])
        return docs, article_scores

    @staticmethod
    def _log_paragraph_distribution(label: str, results: List[dict]) -> None:
        if not results:
            logger.info("[%s] no paragraphs fetched", label)
            return
        per_article = Counter(r["article_id"] for r in results)
        logger.info(
            "[%s] %d paragraphs from %d articles: %s",
            label, len(results), len(per_article),
            ", ".join(f"{art}({n}p)" for art, n in per_article.items()),
        )

    def _rerank_and_decorate(
        self,
        user_query: str,
        docs: List[Document],
        recital_docs: List[Document],
        article_scores: dict,
    ) -> List[Document]:
        for rd in recital_docs:
            rd.metadata["source"] = "recital"
        all_docs = docs + recital_docs

        logger.info(
            "[Retriever] Reranking %d candidates (%d paragraphs + %d recitals)",
            len(all_docs), len(docs), len(recital_docs),
        )

        pairs = [[user_query, doc.page_content] for doc in all_docs]
        ce_scores = self.cross_encoder.predict(pairs)

        boosts = np.array([
            self.recital_score_boost
            if doc.metadata.get("source") == "recital"
            else article_scores.get(doc.metadata.get("article_id"), 0.0) * self.article_score_boost
            for doc in all_docs
        ])
        final_scores = ce_scores + boosts

        ranked_indices = np.argsort(final_scores)[::-1]
        ranked_docs = [all_docs[i] for i in ranked_indices][:self.k]
        ranked_scores = [float(final_scores[i]) for i in ranked_indices][:self.k]

        logger.info(
            "[Retriever] Final top-%d: %s",
            self.k,
            [f"{d.metadata.get('id')}({s:.3f})" for d, s in zip(ranked_docs, ranked_scores)],
        )

        for doc in ranked_docs:
            if doc.metadata.get("source") == "recital":
                header = _recital_header(doc.metadata["celex"], doc.page_content)
                body = _DISPLAY_NUM_RE.sub("", doc.page_content)
                doc.page_content = f"{header}\n{body}"
            else:
                header = _source_header(doc.metadata["article_id"], doc.metadata["article_title"])
                doc.page_content = f"{header}\n{doc.page_content}"

        return ranked_docs

    def _get_relevant_documents(
        self, user_query: str, *, run_manager: CallbackManagerForRetrieverRun = None  # noqa: ARG002
    ) -> List[Document]:
        classification = self.classifier.classify(user_query) if self.classifier else None
        target_acts = classification.acts if classification else []
        query_type = classification.query_type if classification else "ENUMERATION"
        chapter_number = classification.chapter_number if classification else None
        logger.info("[Retriever] target_acts=%s query_type=%s", target_acts, query_type)

        if not target_acts:
            logger.warning("[Retriever] No target acts classified — returning empty.")
            return []

        if query_type == "SCOPE_OF_ACT":
            docs, article_scores = self._fetch_scope_of_act(target_acts)
        elif query_type == "SCOPE_OF_CHAPTER" and chapter_number is not None:
            docs, article_scores = self._fetch_scope_of_chapter(target_acts, chapter_number)
        elif query_type == "DEFINITION_LOOKUP":
            docs, article_scores = self._fetch_definition_lookup(user_query, target_acts)
        elif query_type == "SPECIFIC_QUESTION":
            docs, article_scores = self._fetch_by_semantic_match(
                user_query, target_acts, top_k_articles=self._SPECIFIC_QUESTION_TOP_K
            )
        else:
            docs, article_scores = self._fetch_by_semantic_match(
                user_query, target_acts, top_k_articles=self.top_k_articles
            )

        if not docs:
            logger.warning("[Retriever] %s returned no docs — falling back to semantic match.", query_type)
            docs, article_scores = self._fetch_by_semantic_match(
                user_query, target_acts, top_k_articles=self.top_k_articles
            )

        if not docs:
            return []

        recital_docs = self._match_user_query_to_recitals(user_query, target_acts)
        return self._rerank_and_decorate(user_query, docs, recital_docs, article_scores)
