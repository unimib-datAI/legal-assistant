import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from query import NodeQueries
from service.topic.concept import ConceptService

nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')


class ASKETopicExtractor:
    def __init__(self,
                 graph,
                 model="all-MiniLM-L6-v2"):
        self.graph = graph
        self.embedding_model = SentenceTransformer(model)

    def extract_paragraphs_from_KG(self):
        """Extract paragraphs from Knowledge Graph"""
        paragraphs = []

        with self.graph.driver.session() as session:
            query = NodeQueries.GET_ALL_PARAGRAPHS.format()
            result = session.run(query)

            for record in result:
                paragraphs.append({
                    "paragraph_id": record["paragraph_id"],
                    "text": record["paragraph_text"]
                })

        print(f"Extracted {len(paragraphs)} paragraphs")
        return paragraphs

    def tokenize_paragraphs(self, paragraphs):
        """Tokenize paragraphs into sentences, preserving paragraph_id"""

        tokenized_paragraphs = []
        for para in paragraphs:
            sentences = sent_tokenize(para["text"])
            tokenized_paragraphs.append({
                "paragraph_id": para["paragraph_id"],
                "text": sentences
            })

        print(f"Tokenized {len(tokenized_paragraphs)} paragraphs into sentences")
        return tokenized_paragraphs

    def lemmatize_paragraphs(self, paragraphs):
        """Lemmatize paragraphs using WordNetLemmatizer, preserving paragraph_id

        Accepts either:
        - Raw paragraphs: {"paragraph_id": ..., "text": "string"}
        - Tokenized paragraphs: {"paragraph_id": ..., "text": ["sentence1", "sentence2", ...]}
        """

        lemmatizer = WordNetLemmatizer()
        lemmatized_paragraphs = []

        for para in paragraphs:
            text = para["text"]

            # Tokenized input: lemmatize each sentence
            lemmatized_sentences = []
            for sentence in text:
                words = word_tokenize(sentence)
                lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in words]
                lemmatized_sentences.append(" ".join(lemmatized_words))
            lemmatized_paragraphs.append({
                "paragraph_id": para["paragraph_id"],
                "text": lemmatized_sentences
            })

        print(f"Lemmatized {len(lemmatized_paragraphs)} paragraphs")
        return lemmatized_paragraphs

    def create_chunks_with_paragraph_ids(self, lemmatized_paragraphs, skip_first=True):
        """Create chunks from lemmatized paragraphs, preserving paragraph_id reference.

        Args:
            lemmatized_paragraphs: List of dicts with 'paragraph_id' and 'text' (list of sentences)
            skip_first: If True, skip the first sentence (often just a number/label)

        Returns:
            List of dicts with 'paragraph_id', 'chunk_index', and 'text'
        """
        chunks = []
        for para in lemmatized_paragraphs:
            paragraph_id = para["paragraph_id"]
            sentences = para["text"]

            # Skip first sentence if requested (often just paragraph number)
            start_idx = 1 if skip_first and len(sentences) > 1 else 0

            for chunk_idx, sentence in enumerate(sentences[start_idx:]):
                chunks.append({
                    "paragraph_id": paragraph_id,
                    "chunk_index": chunk_idx,
                    "text": sentence
                })

        print(f"Created {len(chunks)} chunks from {len(lemmatized_paragraphs)} paragraphs")
        return chunks

    def run_aske_cycle(self, chunks, seeds, n_generations=5, alpha=0.4, beta=0.3, gamma=5):
        """Run the full ASKE extraction cycle for N generations.

        Args:
            chunks: List of chunk texts (strings) OR list of dicts with 'paragraph_id' and 'text'
            seeds: Initial seed concepts (list of strings or dicts)
            n_generations: Number of ASKE generations to run
            alpha: Classification similarity threshold
            beta: Terminology enrichment similarity threshold
            gamma: Max new terms per concept per generation

        Returns:
            Tuple of (final concepts, classifications with paragraph_id if available)
        """
        concept_service = ConceptService(self.embedding_model)
        # Initialize concepts from seeds
        print("Initializing concepts from seeds...")
        concepts = self.init_concepts_from_seeds(seeds)

        # Handle both simple string chunks and structured chunks with paragraph_id
        if chunks and isinstance(chunks[0], dict):
            chunk_texts = [c["text"] for c in chunks]
            chunk_metadata = chunks  # Keep full metadata
        else:
            chunk_texts = chunks
            chunk_metadata = None

        # Embed chunks once
        print(f"Embedding {len(chunk_texts)} chunks...")
        chunk_embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=False)

        # Run ASKE generations
        for gen in range(n_generations):
            print(f"\n{'=' * 50}")
            print(f"ASKE Generation {gen + 1}/{n_generations}")
            print(f"{'=' * 50}")

            # Get active concepts
            active_concepts = [current_concept for current_concept in concepts if current_concept.get("active", True)]
            print(f"Active concepts: {len(active_concepts)}")

            if not active_concepts:
                print("No active concepts remaining. Stopping.")
                break

            # Phase 1: Document chunk classification
            print("\n[Phase 1] Document chunk classification...")
            classifications = self.chunk_classification(chunk_texts, chunk_embeddings, active_concepts, alpha, chunk_metadata)
            total_classified = sum(1 for c in classifications if c["concepts"])
            print(f"  Classified {total_classified}/{len(chunk_texts)} chunks")

            # Deactivate unused concepts before terminology enrichment phase
            concepts = concept_service.deactivate_unused_concepts(concepts, classifications)

            # Phase 2: Terminology enrichment
            print("\n[Phase 2] Terminology enrichment...")
            concepts = concept_service.terminology_enrichment(concepts, classifications, chunk_embeddings, chunk_texts, beta=beta, gamma=gamma)

            # Phase 3: Concept derivation (every generation or as configured)
            print("\n[Phase 3] Concept derivation...")
            concepts = concept_service.concept_derivation(concepts)

            # Summary
            active_count = sum(1 for c in concepts if c.get("active", True))
            total_terms = sum(len(c.get("terms", [])) for c in concepts)
            print(f"\nGeneration {gen + 1} summary:")
            print(f"  Total concepts: {len(concepts)}")
            print(f"  Active concepts: {active_count}")
            print(f"  Total terms: {total_terms}")

        return concepts, classifications


    def init_concepts_from_seeds(self, seeds):
        """Initialize concept list from seeds with embeddings."""
        concepts = []

        # Embed each seed and initialize concepts
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

    def chunk_classification(self, chunks, chunk_embeddings, active_concepts, alpha, chunk_metadata=None):
        """Classify individual chunks based on similarity to concept embeddings.

        Args:
            chunks: List of chunk texts
            chunk_embeddings: List of chunk embeddings
            active_concepts: List of concept dictionaries with 'label' and 'embedding'
            alpha: Minimum cosine similarity score to associate a concept (default 0.4)
            chunk_metadata: Optional list of dicts with 'paragraph_id' for each chunk
        """
        classifications = []
        concept_embeddings = np.array([concept["embedding"] for concept in active_concepts])
        concept_labels = [concept["label"] for concept in active_concepts]

        for i, chunk_emb in enumerate(chunk_embeddings):
            similarity_scores  = cosine_similarity(
                [chunk_emb],                    # Shape: (1, embedding_dim)
                concept_embeddings                 # Shape: (n_concepts, embedding_dim)
            )[0]                                   # Extract the single row

            # Find concepts that meet the similarity threshold
            matched_concepts = self._extract_matching_concepts(
                similarity_scores,
                concept_labels,
                threshold=alpha
            )

            classification = {
                "text": chunks[i],
                "concepts": matched_concepts
            }

            # Include paragraph_id if metadata is available
            if chunk_metadata is not None:
                classification["paragraph_id"] = chunk_metadata[i]["paragraph_id"]
                classification["chunk_index"] = chunk_metadata[i].get("chunk_index", i)

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
        # Build list of (label, score) pairs that exceed threshold
        matches = []
        for label, score in zip(concept_labels, similarity_scores):
            if score >= threshold:
                matches.append({
                    "seed": label,
                    "score": float(score)
                })

        # Sort by score (highest first)
        matches.sort(key=lambda x: x["score"], reverse=True)

        # Remove duplicate labels, keeping only the highest score
        deduplicated = []
        seen_labels = set()

        for match in matches:
            label = match["seed"]
            if label not in seen_labels:
                deduplicated.append(match)
                seen_labels.add(label)

        return deduplicated

    def aggregate_topics_by_paragraph(self, classifications, top_n=3, strategy="max"):
        """Aggregate concepts from all chunks of each paragraph and select top N topics.

        When a paragraph is split into multiple chunks, each chunk may have different
        concepts assigned. This method aggregates them using the specified strategy.

        Args:
            classifications: List of classification dicts with 'paragraph_id' and 'concepts'
            top_n: Number of top topics to select per paragraph (default: 3)
            strategy: Aggregation strategy - one of:
                - "max": Take the maximum score for each concept across chunks (default)
                - "avg": Average the scores for each concept
                - "sum": Sum the scores (rewards concepts appearing in multiple chunks)
                - "frequency": Count occurrences, break ties with max score

        Returns:
            Dict mapping paragraph_id to list of top N topics with scores
            Example: {"para_1": [{"topic": "privacy", "score": 0.85}, ...]}
        """
        from collections import defaultdict

        # Group classifications by paragraph_id
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

        # Aggregate concepts for each paragraph
        paragraph_topics = {}

        for paragraph_id, concepts_list in paragraph_concepts.items():
            if not concepts_list:
                paragraph_topics[paragraph_id] = []
                continue

            # Group by topic
            topic_scores = defaultdict(list)
            for c in concepts_list:
                topic_scores[c["topic"]].append(c["score"])

            # Apply aggregation strategy
            aggregated = []
            for topic, scores in topic_scores.items():
                if strategy == "max":
                    final_score = max(scores)
                elif strategy == "avg":
                    final_score = sum(scores) / len(scores)
                elif strategy == "sum":
                    final_score = sum(scores)
                elif strategy == "frequency":
                    # Primary: count, Secondary: max score
                    final_score = len(scores) + max(scores) / 10  # Frequency weighted
                else:
                    raise ValueError(f"Unknown aggregation strategy: {strategy}")

                aggregated.append({
                    "topic": topic,
                    "score": final_score,
                    "chunk_count": len(scores)  # How many chunks had this concept
                })

            # Sort by score and take top N
            aggregated.sort(key=lambda x: x["score"], reverse=True)
            paragraph_topics[paragraph_id] = aggregated[:top_n]

        print(f"Aggregated topics for {len(paragraph_topics)} paragraphs using '{strategy}' strategy")
        return paragraph_topics

    def update_paragraph_topics(self, paragraph_topics: dict[str, list[dict]]) -> int:
        """Create Topic nodes and RELATED_TO relationships from paragraphs.

        For each paragraph, creates Topic nodes (if not existing) and links
        them via (Paragraph)-[:RELATED_TO]->(Topic).

        Args:
            paragraph_topics: Dict mapping paragraph_id to list of topic dicts
                e.g. {"para_1": [{"topic": "privacy", "score": 0.85}, ...]}

        Returns:
            Number of paragraphs linked to topics
        """
        updated_count = 0
        created_topics: set[str] = set()

        with self.graph.driver.session() as session:
            for paragraph_id, topics in paragraph_topics.items():
                for topic_dict in topics:
                    topic_label = topic_dict["topic"]

                    # Create Topic node once per unique label
                    if topic_label not in created_topics:
                        session.run(
                            NodeQueries.CREATE_TOPIC_NODE,
                            topic_label=topic_label,
                        )
                        created_topics.add(topic_label)

                    # Create RELATED_TO relationship
                    session.run(
                        NodeQueries.CREATE_PARAGRAPH_TOPIC_RELATIONSHIP,
                        paragraph_id=paragraph_id,
                        topic_label=topic_label,
                    )

                updated_count += 1

        print(f"Created {len(created_topics)} Topic nodes")
        print(f"Linked {updated_count} paragraphs to topics via RELATED_TO")
        return updated_count


