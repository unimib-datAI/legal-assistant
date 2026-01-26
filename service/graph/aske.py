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
                    "text": record["paragraph_text"]
                })

        print(f"Extracted {len(paragraphs)} paragraphs")
        return paragraphs

    def tokenize_paragraphs(self, paragraphs):
        """Tokenize paragraphs into sentences"""

        tokenized_paragraphs = []
        for para in paragraphs:
            sentences = sent_tokenize(para["text"])
            tokenized_paragraphs.append({
                "text": sentences
            })

        print(f"Tokenized {len(tokenized_paragraphs)} paragraphs into sentences")
        return tokenized_paragraphs

    def lemmatize_paragraphs(self, paragraphs):
        """Lemmatize paragraphs using WordNetLemmatizer

        Accepts either:
        - Raw paragraphs: {"id": ..., "text": "string"}
        - Tokenized paragraphs: {"id": ..., "text": ["sentence1", "sentence2", ...]}
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
                "text": lemmatized_sentences
            })

        print(f"Lemmatized {len(lemmatized_paragraphs)} paragraphs")
        return lemmatized_paragraphs


    def run_aske_cycle(self, chunks, seeds, n_generations=5, alpha=0.4, beta=0.3, gamma=5):
        """Run the full ASKE extraction cycle for N generations.

        Args:
            chunks: List of document chunk texts
            seeds: Initial seed concepts (list of strings or dicts)
            n_generations: Number of ASKE generations to run
            alpha: Classification similarity threshold
            beta: Terminology enrichment similarity threshold
            gamma: Max new terms per concept per generation

        Returns:
            Final concepts after N generations
        """
        concept_service = ConceptService(self.embedding_model)
        # Initialize concepts from seeds
        print("Initializing concepts from seeds...")
        concepts = self.init_concepts_from_seeds(seeds)

        # Embed chunks once
        print(f"Embedding {len(chunks)} chunks...")
        chunk_embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)

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
            classifications = self.chunk_classification(chunks, chunk_embeddings, active_concepts, alpha)
            total_classified = sum(1 for c in classifications if c["concepts"])
            print(f"  Classified {total_classified}/{len(chunks)} chunks")

            # Deactivate unused concepts before terminology enrichment phase
            concepts = concept_service.deactivate_unused_concepts(concepts, classifications)

            # Phase 2: Terminology enrichment
            print("\n[Phase 2] Terminology enrichment...")
            concepts = concept_service.terminology_enrichment(concepts, classifications, chunk_embeddings, chunks, beta=beta, gamma=gamma)

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

    def chunk_classification(self, chunks, chunk_embeddings, active_concepts, alpha):
        """Classify individual chunks based on similarity to concept embeddings.

        Args:
            chunks: List of chunk texts
            chunk_embeddings: List of chunk embeddings
            active_concepts: List of concept dictionaries with 'label' and 'embedding'
            alpha: Minimum cosine similarity score to associate a concept (default 0.4)
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

            classifications.append({
                "text": chunks[i],
                "concepts": matched_concepts
            })
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


