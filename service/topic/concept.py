import nltk
import warnings
from collections import defaultdict

import nltk
import numpy as np
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from sklearn.cluster import AffinityPropagation
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

class ConceptService:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def terminology_enrichment(self,
                               concepts,
                               classifications,
                               chunk_embeddings,
                               chunks,
                               beta=0.3,
                               gamma=5,
                               max_candidates=50):
        """Extract new terms from classified chunks and enrich concept terminology.

        Args:
            concepts: List of concept dictionaries with 'label', 'embedding', 'terms', 'active'
            classifications: List of classification results from document_chunk_classification
            chunk_embeddings: Pre-computed chunk embeddings (numpy array)
            chunks: List of chunk texts
            beta: Minimum similarity threshold for term enrichment (default 0.3)
            gamma: Maximum number of new terms per concept per generation (default 5)
            max_candidates: Maximum candidate terms to evaluate per concept (default 50)

        Returns:
            Updated concepts with enriched terminology
        """
        # Initialize a dictionary to map concept to every label found in classification "text" field
        concept_chunk_indices = defaultdict(list)
        for i, classification in enumerate(classifications):
            for matched in classification["concepts"]:
                concept_label = matched["seed"]
                concept_chunk_indices[concept_label].append(i)

        # For each active concept, extract and evaluate new terms
        for current_concept in concepts:
            # skip current inactive concept
            if not current_concept.get("active", True):
                continue

            # Extract all chunks indexes classified under this concept. "concept1" -> [chunk_idx1, chunk_idx2, ...]
            label_concept = current_concept["label"]
            chunk_indices = concept_chunk_indices.get(label_concept, [])

            if not chunk_indices:
                continue

            # Get chunks for this concept
            concept_chunks = [chunks[i] for i in chunk_indices[:100]]  # to Limit chunks -> [:100]
            # Extract candidate terms (limit to most frequent)
            candidate_terms = self._extract_candidate_terms(concept_chunks, current_concept, max_candidates=max_candidates)

            if not candidate_terms:
                continue

            # Get WordNet definitions (only first/most common sense for speed)
            term_definitions = self._extract_wordnet_definitions(candidate_terms)

            if not term_definitions:
                continue

            # BATCH embed all definitions at once
            all_definitions = [td["definition"] for td in term_definitions]
            def_embeddings = self.embedding_model.encode(all_definitions, show_progress_bar=False)

            # Evaluate enriched terms
            enriched_terms = self._evaluate_enriched_terms(
                current_concept, chunk_indices, chunk_embeddings, term_definitions, def_embeddings, beta
            )
            # Sort by score and apply learning rate gamma
            enriched_terms.sort(key=lambda x: x["score"], reverse=True)
            new_terms = enriched_terms[:gamma]

            # Add new terms to concept
            if "terms" not in current_concept:
                current_concept["terms"] = []
            current_concept["terms"].extend(new_terms)

            if new_terms:
                print(f"  Concept '{label_concept}': +{len(new_terms)} terms")

        return concepts

    def concept_derivation(self, concepts, min_terms_for_clustering=3):
        """Derive new concepts by clustering terms within each concept.

        Uses Affinity Propagation to cluster term embeddings and create new concepts.

        Args:
            concepts: List of concept dictionaries
            min_terms_for_clustering: Minimum terms needed to attempt clustering (default 3)

        Returns:
            Updated list of concepts with newly derived concepts
        """
        new_concepts = []

        # Collect terms that need embedding
        terms_to_embed = []
        terms_to_embed_info = []  # (concept_idx, term_idx, term_label)

        for c_idx, concept in enumerate(concepts):
            if not concept.get("active", True):
                continue
            for t_idx, term in enumerate(concept.get("terms", [])):
                if isinstance(term, str):
                    terms_to_embed.append(term)
                    terms_to_embed_info.append((c_idx, t_idx, term))

        # Batch embed missing terms
        if terms_to_embed:
            print(f"  Embedding {len(terms_to_embed)} terms...")
            new_embeddings = self.embedding_model.encode(terms_to_embed, show_progress_bar=False)
            for i, (c_idx, t_idx, label) in enumerate(terms_to_embed_info):
                concepts[c_idx]["terms"][t_idx] = {"label": label, "embedding": new_embeddings[i]}

        for concept in concepts:
            if not concept.get("active", True):
                new_concepts.append(concept)
                continue

            terms = concept.get("terms", [])

            # Need minimum terms to cluster meaningfully
            if len(terms) < min_terms_for_clustering:
                new_concepts.append(concept)
                continue

            # Get term embeddings
            term_embeddings = []
            term_labels = []
            for term in terms:
                if isinstance(term, dict) and "embedding" in term:
                    term_embeddings.append(term["embedding"])
                    term_labels.append(term["label"])

            if len(term_embeddings) < min_terms_for_clustering:
                new_concepts.append(concept)
                continue

            term_embeddings = np.array(term_embeddings)

            # Apply Affinity Propagation clustering
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    clustering = AffinityPropagation(
                        random_state=42,
                        damping=0.9,  # Higher damping = more stable
                        max_iter=300,
                        convergence_iter=30
                    )
                    cluster_labels = clustering.fit_predict(term_embeddings)

                # Check if clustering produced valid results
                if len(set(cluster_labels)) <= 1 or -1 in cluster_labels:
                    # Clustering failed or produced single cluster
                    new_concepts.append(concept)
                    continue
            except Exception as e:
                print(f"  Clustering failed for '{concept['label']}': {e}")
                new_concepts.append(concept)
                continue

            # Group terms by cluster
            clusters = defaultdict(list)
            for i, cluster_id in enumerate(cluster_labels):
                clusters[cluster_id].append({
                    "label": term_labels[i],
                    "embedding": term_embeddings[i],
                    "term_data": terms[i] if i < len(terms) else None
                })

            # If only 1 cluster, keep original concept
            if len(clusters) <= 1:
                new_concepts.append(concept)
                continue

            # Create concepts from clusters
            original_label = concept["label"]

            for cluster_id, cluster_terms in clusters.items():
                cluster_embs = np.array([t["embedding"] for t in cluster_terms])
                centroid = np.mean(cluster_embs, axis=0)

                # Find term closest to centroid
                distances = cosine_similarity([centroid], cluster_embs)[0]
                closest_idx = np.argmax(distances)
                new_label = cluster_terms[closest_idx]["label"]

                # Check if this cluster conserves the original concept
                is_conservation = new_label == original_label or original_label.lower() in [t["label"].lower() for t in
                                                                                            cluster_terms]

                new_concept = {
                    "label": new_label if not is_conservation else original_label,
                    "embedding": centroid,
                    "terms": [t.get("term_data", {"label": t["label"]}) for t in cluster_terms],
                    "active": True,
                    "derived_from": None if is_conservation else original_label,
                    "generation": concept.get("generation", 0) + (0 if is_conservation else 1)
                }

                new_concepts.append(new_concept)

            print(f"  '{original_label}' -> {len(clusters)} concepts")

        # Deduplication check
        deduplicated = self._deduplicate_concepts(new_concepts)

        return deduplicated

    def deactivate_unused_concepts(self, concepts, classifications):
        """Deactivate concepts that don't classify any document chunk.

        Args:
            concepts: List of concept dictionaries
            classifications: Classification results

        Returns:
            Updated concepts with inactive flags
        """
        # Count chunks per concept
        concept_chunk_counts = defaultdict(int)
        for classification in classifications:
            for matched in classification["concepts"]:
                label = matched["seed"]
                concept_chunk_counts[label] += 1

        # Deactivate concepts with no chunks
        deactivated_count = 0
        for concept in concepts:
            label = concept["label"]
            is_active_concept = concept.get("active", True)
            if concept_chunk_counts[label] == 0 and is_active_concept:
                concept["active"] = False
                deactivated_count += 1

        if deactivated_count > 0:
            print(f"  Deactivated {deactivated_count} concepts with no classified chunks")

        return concepts

    def _extract_candidate_terms(self, concept_chunks, current_concept, max_candidates=50):
        """
        Extract candidate terms from concept chunks based on term frequency.
        """
        term_counts = defaultdict(int)
        for chunk in concept_chunks:
            words = word_tokenize(chunk)  # Tokenize chunk into words
            for word in words:
                if word.isalpha() and len(word) > 3:  # no number, just letter and increased min length
                    term_counts[word.lower()] += 1  # Count term frequency

        # Extract existing terms from the concept
        existing_terms = set()
        for term in current_concept.get("terms", []):
            label = term["label"] if isinstance(term, dict) else term
            existing_terms.add(label)

        # Filter out terms that already exist
        candidate_terms = []
        for term, count in term_counts.items():
            if term not in existing_terms:
                candidate_terms.append((term, count))

        # Sort by frequency (highest first) and keep top candidates
        candidate_terms.sort(key=lambda x: x[1], reverse=True)
        candidate_terms = [term for term, count in candidate_terms[:max_candidates]]

        return candidate_terms

    def _extract_wordnet_definitions(self, candidate_terms):
        """
        Extract WordNet definitions for candidate terms.
        """
        term_definitions = []
        for term in candidate_terms:
            synsets = wordnet.synsets(term)
            if synsets:
                # Use only the first (most common) definition for speed
                term_definitions.append({
                    "label": term,
                    "definition": synsets[0].definition()
                })
        return term_definitions

    def _evaluate_enriched_terms(self, current_concept, chunk_indices, chunk_embeddings, term_definitions, def_embeddings, beta):
        """
        Compute the centroid of chunk embeddings for the current concept:
        Chunk 1 embedding: [0.2, 0.5, 0.1]
        Chunk 2 embedding: [0.3, 0.4, 0.2]
        Chunk 3 embedding: [0.1, 0.6, 0.0]
        ─────────────────────────────────
        Centroid:         [0.2, 0.5, 0.1]  ← column avg
        """
        chunk_embs = chunk_embeddings[chunk_indices[:100]]
        chunk_centroid = np.mean(chunk_embs, axis=0)
        concept_emb = current_concept["embedding"]

        # Evaluate all terms
        enriched_terms = []
        for i, term_def in enumerate(term_definitions):
            term_emb = def_embeddings[i]

            # Compute combined similarity score
            sim_to_concept = float(cosine_similarity([concept_emb], [term_emb])[0][0])
            sim_to_chunks = float(cosine_similarity([term_emb], [chunk_centroid])[0][0])
            combined_score = sim_to_concept + sim_to_chunks

            if combined_score >= beta:
                enriched_terms.append({
                    "label": term_def["label"],
                    "definition": term_def["definition"],
                    "embedding": term_emb,
                    "score": combined_score
                })

        return enriched_terms

    def _deduplicate_concepts(self, concepts):
        """Remove duplicate concepts by label and by term sets."""
        # Sort by generation (older first) to keep older concepts
        concepts_sorted = sorted(concepts, key=lambda c: c.get("generation", 0))

        kept_concepts = []
        seen_labels = set()
        seen_term_sets = []

        for concept in concepts_sorted:
            label = concept["label"]

            # Check for duplicate label
            if label in seen_labels:
                print(f"  Discarding duplicate label '{label}'")
                continue

            term_labels = set(
                t["label"] if isinstance(t, dict) else t
                for t in concept.get("terms", [])
            )

            # Check if this term set is a subset of any existing concept
            is_duplicate = False
            for seen_terms in seen_term_sets:
                if term_labels and term_labels.issubset(seen_terms):
                    is_duplicate = True
                    print(f"  Discarding subset concept '{label}'")
                    break

            if not is_duplicate:
                kept_concepts.append(concept)
                seen_labels.add(label)
                if term_labels:
                    seen_term_sets.append(term_labels)

        return kept_concepts