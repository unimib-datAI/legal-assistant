import logging
import re
from typing import List, Any

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from pydantic import ConfigDict
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

from service.graph.query import NodeQueries

logger = logging.getLogger(__name__)


class GraphEnrichedRetriever(BaseRetriever):
    """Retriever combining semantic topic filtering with vector similarity search."""
    vector_store: Any
    graph: Any
    k: int = 5
    use_topic_filter: bool = True
    top_k_topic: int = 5
    topic_similarity_threshold: float = 0.35
    embedding_model: Any = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder: Any = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    graph_topic: dict = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def extract_unique_topics_and_related_embeddings(self) -> dict:
        """Extract the graph UNIQUE topic for every tagged paragraph, embed them and save it in the next steps."""

        if self.graph_topic is not None:
            return self.graph_topic

        result = self.graph.query(NodeQueries.GET_ALL_UNIQUE_TOPICS)
        extracted_topics = result[0]["topics"]

        if not extracted_topics:
            self.graph_topic = {"topics": [], "embeddings": np.array([])}
        else:
            embeddings = self.embedding_model.encode(extracted_topics, show_progress_bar=False)
            self.graph_topic = {"topics": extracted_topics, "embeddings": np.array(embeddings)}

        return self.graph_topic

    def _filter_and_rank_topics(self, topics: List[str], scores: np.ndarray) -> List[tuple]:
        """Filter topics by threshold and return top-k sorted by score DESC."""
        matches = []

        for topic, score in zip(topics, scores):
            if score >= self.topic_similarity_threshold:
                matches.append((topic, float(score)))
        return matches

    def _match_topics(self, user_query: str) -> List[tuple]:
        """Find topics semantically similar to the user query."""
        extracted_topics = self.extract_unique_topics_and_related_embeddings()

        if not extracted_topics["topics"]:
            logger.info("No topics found for user query: %s", user_query)
            return []

        query_embedding = self.embedding_model.encode(user_query, show_progress_bar=False)
        similarities = cosine_similarity([query_embedding], extracted_topics["embeddings"])[0]

        matches = self._filter_and_rank_topics(extracted_topics["topics"], similarities)
        matches.sort(key=lambda match: match[1], reverse=True)

        return matches[:self.top_k_topic]

    def _get_paragraphs_by_topics(self, user_query:str, topics: List[str]) -> List[Document]:
        """Retrieve paragraphs associated with the given topics."""
        if not topics:
            return []

        results = self.graph.query(
            NodeQueries.GET_ALL_PARAGRAPHS_BY_TOPIC,
            params={"topics": topics}
        )
        extracted_paragraphs = self._get_document_from_query_result(results)
        ranked_docs = self._rank_extracted_documents(user_query, extracted_paragraphs, 3)

        return ranked_docs

    @staticmethod
    def _extract_paragraph_id(content: str) -> str:
        """Extract paragraph ID from document content."""
        match = re.search(r'\nid:\s*(\S+)', content)
        return match.group(1) if match else None

    def _get_relevant_documents(self, user_query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        seen_ids = set()
        relevant_docs = []

        if self.use_topic_filter:
            matched_topics = self._match_topics(user_query)

            if matched_topics:
                logger.info("[Semantic Topic Filter] Matched topics: %s",
                            ", ".join(f"{t}({s:.3f})" for t, s in matched_topics))

                topic_names = [t for t, _ in matched_topics]
                topic_docs = []
                for curr_doc in self._get_paragraphs_by_topics(user_query, topic_names):
                    paragraph_id = curr_doc.metadata.get("id")
                    if paragraph_id and paragraph_id not in seen_ids:
                        seen_ids.add(paragraph_id)
                        relevant_docs.append(curr_doc)
                        topic_docs.append(paragraph_id)
                logger.info("[Topic Filter] Retrieved %d paragraphs: %s", len(topic_docs), topic_docs)

        vector_ids = []
        for curr_doc in self.vector_store.similarity_search(user_query, k=self.k * 2):
            paragraph_id = curr_doc.metadata.get("id") or self._extract_paragraph_id(curr_doc.page_content)
            if paragraph_id not in seen_ids:
                curr_doc.metadata["id"] = paragraph_id
                curr_doc.metadata["source"] = "vector_search"
                seen_ids.add(paragraph_id)
                relevant_docs.append(curr_doc)
                vector_ids.append(paragraph_id)
        logger.info("[Vector Search] Retrieved %d paragraphs: %s", len(vector_ids), vector_ids)

        logger.info("[Retriever] Reranking %d documents total", len(relevant_docs))
        if not relevant_docs:
            return []

        pairs = [[user_query, doc.page_content] for doc in relevant_docs]
        scores = self.cross_encoder.predict(pairs)

        ranked_indices = np.argsort(scores)[::-1]
        ranked_docs = [relevant_docs[i] for i in ranked_indices][:self.k * 2]

        logger.info("[Retriever] Final top-%d: %s", self.k, [d.metadata.get("id") for d in ranked_docs])
        return ranked_docs


    def _rank_extracted_documents(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        """Rank extracted document candidates from the topic extraction from the user query using BM25 ranker."""
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = top_k
        results = bm25_retriever.invoke(query)
        for r in results:
            logger.info("[BM25 Ranker] ranked doc content: %s and metadata: %s", r.page_content, r.metadata)

        return results

    def _get_document_from_query_result(self, query_result):
        return [
            Document(
                page_content=f"{r['text']}",
                metadata={
                    "id": r["id"],
                    "topics": r["topics"],
                    "article_title": r["article_title"],
                    "source": "semantic_topic_filter"
                }
            )
            for r in query_result
        ]


