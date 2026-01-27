
import config
from service.graph.aske import ASKETopicExtractor
from service.graph.graph import Neo4jGraph
from service.graph.seed import SEEDS_AI_DATA_FOCUSED

graph = Neo4jGraph(config.NEO4J_URI, config.NEO4J_USERNAME, config.NEO4J_PASSWORD)

ASKE = ASKETopicExtractor(graph)

# Extract and preprocess paragraphs
paragraphs = ASKE.extract_paragraphs_from_KG()
tokenized_paragraphs = ASKE.tokenize_paragraphs(paragraphs)
lemmatized_paragraphs = ASKE.lemmatize_paragraphs(tokenized_paragraphs)

# Create chunks with paragraph_id references (skipping first element which is the number)
chunks = ASKE.create_chunks_with_paragraph_ids(lemmatized_paragraphs, skip_first=True)

print(f"\nTotal chunks extracted: {len(chunks)}")
print(f"Sample chunk: {chunks[0]['text'][:100]}...")

# Run full ASKE cycle for N generations
N_GENERATIONS = 15      # Number of ASKE generations
ALPHA = 0.3             # Classification threshold
BETA = 0.3              # Terminology enrichment threshold
GAMMA = 10              # Max new terms per concept per generation

# Select the seed set to use
test_seeds = SEEDS_AI_DATA_FOCUSED

print(f"\nStarting ASKE cycle with {len(test_seeds)} seed concepts...")
print(f"Parameters: alpha={ALPHA}, beta={BETA}, gamma={GAMMA}, generations={N_GENERATIONS}")

concepts, final_classifications = ASKE.run_aske_cycle(
    chunks=chunks,
    seeds=test_seeds,
    n_generations=N_GENERATIONS,
    alpha=ALPHA,
    beta=BETA,
    gamma=GAMMA
)

# Print final results
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

# Active concepts
active_concepts = [c for c in concepts if c.get("active", True)]
inactive_concepts = [c for c in concepts if not c.get("active", True)]

print(f"\nTotal concepts: {len(concepts)}")
print(f"Active concepts: {len(active_concepts)}")
print(f"Inactive (deactivated) concepts: {len(inactive_concepts)}")

# Print active concepts with their terms
print("\n--- Active Concepts ---")
for concept in active_concepts[:20]:  # Show first 20
    terms = concept.get("terms", [])
    term_labels = [t["label"] if isinstance(t, dict) else t for t in terms]
    derived = f" (derived from: {concept['derived_from']})" if concept.get("derived_from") else ""
    print(f"\n  [{concept['label']}]{derived}")
    print(f"    Generation: {concept.get('generation', 0)}")
    print(f"    Terms ({len(term_labels)}): {', '.join(term_labels[:10])}{'...' if len(term_labels) > 10 else ''}")

# Print deactivated concepts
if inactive_concepts:
    print("\n--- Deactivated Concepts ---")
    for concept in inactive_concepts[:10]:
        print(f"  - {concept['label']}")

# Print sample classifications
print("\n--- Sample Classifications ---")
for i, result in enumerate(final_classifications[:5]):
    if result["concepts"]:
        print(f"\nChunk {i+1}: {result['text'][:80]}...")
        print(f"  Paragraph ID: {result.get('paragraph_id', 'N/A')}")
        print(f"  Matched concepts:")
        for c in result["concepts"][:3]:
            print(f"    - {c['seed']} (score: {c['score']:.3f})")

# Aggregate topics by paragraph and update Neo4j
print("\n" + "="*60)
print("AGGREGATING TOPICS BY PARAGRAPH")
print("="*60)

# Aggregate using max score strategy, top 3 topics per paragraph
paragraph_topics = ASKE.aggregate_topics_by_paragraph(
    final_classifications,
    top_n=3,
    strategy="max"
)

# Update Neo4j with topics
print("\n--- Updating Neo4j ---")
updated_count = ASKE.update_paragraph_topics(paragraph_topics)
print(f"Updated topics for {updated_count} paragraphs in Neo4j.")