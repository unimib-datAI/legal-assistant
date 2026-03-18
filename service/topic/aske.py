import logging
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from service.topic.concept import ConceptService


"""
Paper reference: https://doi.org/10.1016/j.clsr.2023.105903

ASKE (Automated Knowledge Extraction) is an iterative process to extract and refine topics from a corpus using a seed-based approach.
1. Start with initial seed concepts (e.g., "privacy", "data protection").
2. Classify document chunks by similarity to concept embeddings (Phase 1).
3. Enrich concepts with new terms found in classified chunks (Phase 2).
4. Derive new concepts from enriched terms (Phase 3).
Repeat for N generations, deactivating concepts that are no longer relevant.
"""

logger = logging.getLogger(__name__)

class ASKETopicExtractor:
    def __init__(self, graph, model="all-MiniLM-L6-v2"):
        self.graph = graph
        self.embedding_model = SentenceTransformer(model)

    def run_aske_cycle(self, chunks, seeds, n_generations=5, alpha=0.4, beta=0.3, gamma=5):
        """Run the full ASKE extraction cycle for N generations.

        Args:
            chunks: List of chunks to classify, it contains a dict with the chunk information (such as text, id and
                    chunk_idx if it refers to a part of a long text that was split into multiple chunks).
            seeds: Initial seed concepts (list of strings or dicts)
            n_generations: Number of ASKE generations to run
            alpha: Classification similarity threshold
            beta: Terminology enrichment similarity threshold
            gamma: Max new terms per concept per generation

        Returns:
            Tuple of (final concepts, classifications)
        """
        logger.info("Initializing %d seed concepts", len(seeds))
        concepts = self._init_concepts_from_seeds(seeds)
        seed_embeddings, seed_labels = self._extract_concept_information(concepts)
        concept_service = ConceptService(self.embedding_model, seed_embeddings=seed_embeddings, seed_labels=seed_labels)

        # Handle both simple string chunks and structured chunks with paragraph_id
        chunks_text, chunks_metadata = self._extract_chunk_information(chunks)

        logger.info("Embedding %d chunks...", len(chunks_text))
        chunks_embeddings = self.embedding_model.encode(chunks_text, show_progress_bar=False)

        classifications = []
        for gen in range(n_generations):
            active_concepts = [concept for concept in concepts if concept.get("active", True)]

            if not active_concepts:
                logger.info("No active concepts remaining. Stopping at generation %d.", gen + 1)
                break

            # Phase 1: Document chunk classification
            classifications = self._chunk_classification(chunks_text, chunks_embeddings, chunks_metadata, active_concepts, alpha)
            # Deactivate unused concepts before terminology enrichment phase
            concepts = concept_service.deactivate_unused_concepts(concepts, classifications)
            # Phase 2: Terminology enrichment
            concepts = concept_service.terminology_enrichment(concepts, classifications, chunks_embeddings, chunks_text, beta=beta, gamma=gamma)
            # Phase 3: Concept derivation
            concepts = concept_service.concept_derivation(concepts)

            active_count = sum(1 for active_concept in concepts if active_concept.get("active", True))
            logger.info("Generation %d, Total concepts: %d, Active concepts: %d", gen + 1, len(concepts), active_count)

        return concepts, classifications


    def _chunk_classification(self, chunks_text, chunks_embeddings, chunk_metadata, active_concepts, alpha):
        """Classify individual chunks based on similarity to concept embeddings.

        Args:
            chunks_text: List of chunk texts
            chunks_embeddings: List of chunk embeddings
            active_concepts: List of concept dictionaries with 'label' and 'embedding'
            alpha: Minimum cosine similarity score to associate a concept (default 0.4)
            chunk_metadata: Optional list of dicts with 'paragraph_id' for each chunk
        """
        classifications = []
        concept_embeddings = np.array([concept["embedding"] for concept in active_concepts])
        concept_labels = [concept["label"] for concept in active_concepts]

        for idx, chunk_emb in enumerate(chunks_embeddings):
            similarity_scores = cosine_similarity(
                [chunk_emb],           # Shape: (1, embedding_dim)
                concept_embeddings     # Shape: (n_concepts, embedding_dim)
            )[0]                       # Extract the single row

            matched_concepts = self._extract_matching_concepts(
                similarity_scores,
                concept_labels,
                threshold=alpha
            )

            classification = {
                "chunk_index": chunk_metadata[idx].get("chunk_index", idx),
                "paragraph_id": chunk_metadata[idx]["paragraph_id"],
                "text": chunks_text[idx],
                "concepts": matched_concepts
            }

            classifications.append(classification)
        return classifications

    @staticmethod
    def _extract_matching_concepts(similarity_scores, concept_labels, threshold):
        """
        Extract concepts that exceed the similarity threshold.

        Helper function that:
        1. Filters concepts by minimum similarity threshold
        2. Handles duplicate concept labels (keeps highest score)
        3. Sorts results by similarity score (descending)

        Args:
            similarity_scores: Array of similarity scores for each concept
            concept_labels: List of concept names (parallel to similarity_scores)
            threshold: Minimum similarity score to include

        Returns:
            List of matched concepts with their scores, sorted by score descending.
            Each entry is {'concept': str, 'score': float}
        """
        matches = []
        for label, score in zip(concept_labels, similarity_scores):
            if score >= threshold:
                matches.append({
                    "seed": label,
                    "score": float(score)
                })

        matches.sort(key=lambda x: x["score"], reverse=True)

        deduplicated = []
        seen_labels = set()

        for match in matches:
            label = match["seed"]
            if label not in seen_labels:
                deduplicated.append(match)
                seen_labels.add(label)

        return deduplicated

    @staticmethod
    def aggregate_topics_by_paragraph(classifications, top_n=3):
        """Aggregate concepts from all chunks of each paragraph and select top N topics.

        When a paragraph is split into multiple chunks, each chunk may have different
        concepts assigned. This method aggregates them using the specified strategy.

        Args:
            classifications: List of classification dicts with 'paragraph_id' and 'concepts'
            top_n: Number of top topics to select per paragraph (default: 3)

        Returns:
            Dict mapping paragraph_id to list of top N topics with scores
            Example: {"para_1": [{"topic": "privacy", "score": 0.85}, ...]}
        """
        paragraph_concepts = defaultdict(list)

        for classification in classifications:
            paragraph_id = classification.get("paragraph_id")

            if paragraph_id is None:
                continue

            for concept in classification.get("concepts", []):
                paragraph_concepts[paragraph_id].append({
                    "topic": concept["seed"],
                    "score": concept["score"]
                })

        paragraph_topics = {}

        for paragraph_id, concepts_list in paragraph_concepts.items():
            if not concepts_list:
                paragraph_topics[paragraph_id] = []
                continue

            topic_scores = defaultdict(list)
            for c in concepts_list:
                topic_scores[c["topic"]].append(c["score"])

            aggregated = []
            for topic, scores in topic_scores.items():
                aggregated.append({
                    "topic": topic,
                    "score": max(scores),
                    "chunk_count": len(scores)
                })

            aggregated.sort(key=lambda x: x["score"], reverse=True)
            paragraph_topics[paragraph_id] = aggregated[:top_n]

        logger.info("Aggregated topics for %d paragraphs", len(paragraph_topics))
        return paragraph_topics

    def _init_concepts_from_seeds(self, seeds):
        """Initialize concept list from seeds with embeddings."""
        concepts = []

        for seed in seeds:
            seed_embedding = self.embedding_model.encode(seed, show_progress_bar=False)
            concepts.append({
                "label": seed,
                "embedding": seed_embedding,
                "terms": [{
                    "label": seed,
                    "embedding": seed_embedding
                }],
                "active": True,
                "generation": 0
            })

        return concepts

    @staticmethod
    def _extract_concept_information(concepts):
        """Extract embeddings and labels from concept dictionaries."""
        seed_embeddings = np.array([concept["embedding"] for concept in concepts])
        seed_labels = [concept["label"] for concept in concepts]

        return seed_embeddings, seed_labels

    @staticmethod
    def _extract_chunk_information(chunks):
        """Extract text and metadata from chunk dictionaries."""
        chunks_text = [chunk["text"] for chunk in chunks]
        chunks_metadata = chunks

        return chunks_text, chunks_metadata
