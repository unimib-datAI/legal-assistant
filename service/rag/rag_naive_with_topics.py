import re
import logging
import numpy as np

from typing import List, Any

from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pydantic import ConfigDict
from sentence_transformers import SentenceTransformer
from service.graph.query import NodeQueries

class GraphEnrichedRetriever(BaseRetriever):
    """Retriever combining semantic topic filtering with vector similarity search."""
    vector_store: Any
    graph: Any
    k: int = 5
    use_topic_filter: bool = True
    top_k_topic: int = 5
    topic_similarity_threshold: float = 0.35
    embedding_model: Any = SentenceTransformer("all-MiniLM-L6-v2")
    graph_topic: dict = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def extract_unique_topics_and_related_embeddings(self) -> dict:
        """Extract the graph UNIQUE topic for every tagged paragraph, embed them and save it in the next steps."""

        if self.graph_topic is not None:
            return self.graph_topic

        result = self.graph.query(NodeQueries.GET_ALL_UNIQUE_TOPICS)
        extracted_topics = result[0]["topics"]

        if not extracted_topics:
            # return empty for consistency
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
            logging.info(f"No topics found for user query: {user_query}")
            return []

        # Generate embedding for the user query to match against topic embeddings
        query_embedding = self.embedding_model.encode(user_query, show_progress_bar=False)
        similarities = cosine_similarity([query_embedding], extracted_topics["embeddings"])[0]

        matches = self._filter_and_rank_topics(extracted_topics["topics"], similarities)
        matches.sort(key=lambda match: match[1], reverse=True)

        # Return top-k topics
        return matches[:self.top_k_topic]

    def _get_paragraphs_by_topics(self, topics: List[str]) -> List[Document]:
        """Retrieve paragraphs associated with the given topics."""
        if not topics:
            return []

        results = self.graph.query(
            NodeQueries.GET_ALL_PARAGRAPHS_BY_TOPIC,
            params={"topics": topics, "limit": self.k * 2}
        )

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
            for r in results
        ]

    @staticmethod
    def _extract_paragraph_id(content: str) -> str:
        """Extract paragraph ID from document content."""
        match = re.search(r'\nid:\s*(\S+)', content)
        return match.group(1) if match else None

    def _get_relevant_documents(self, user_query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        seen_ids = set()
        relevant_docs = []

        # Filter the KG based on semantic topic matching
        if self.use_topic_filter:
            matched_topics = self._match_topics(user_query)

            if matched_topics:
                logging.info("[Semantic Topic Filter] Matched topics:")
                for topic, score in matched_topics:
                    logging.info(f"  - {topic}: {score:.3f}")

                topic_names = [t for t, _ in matched_topics]
                for curr_doc in self._get_paragraphs_by_topics(topic_names):
                    paragraph_id = curr_doc.metadata.get("id")
                    if paragraph_id and paragraph_id not in seen_ids:
                        seen_ids.add(paragraph_id)
                        relevant_docs.append(curr_doc)

        # Vector similarity search
        for curr_doc in self.vector_store.similarity_search(user_query, k=self.k * 2):
            paragraph_id = curr_doc.metadata.get("id")
            if paragraph_id not in seen_ids:
                curr_doc.metadata["id"] = paragraph_id
                curr_doc.metadata["source"] = "vector_search"
                seen_ids.add(paragraph_id)
                relevant_docs.append(curr_doc)

        # Rerank relevant docs
        logging.info(f"[Retriever] Reranking {len(relevant_docs)} documents")
        query_embedding = self.embedding_model.encode(user_query, show_progress_bar=False)
        doc_embeddings = self.embedding_model.encode(
            [doc.page_content for doc in relevant_docs],
            show_progress_bar=False
        )
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

        ranked_indices = np.argsort(similarities)[::-1] # Descending order
        ranked_docs = [relevant_docs[i] for i in ranked_indices]

        return ranked_docs[:self.k]