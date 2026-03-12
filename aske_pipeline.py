import logging
import pathlib

import config
from service.graph.graph import Neo4jGraph
from service.graph.seed import SEEDS_AI_DATA_FOCUSED_v2
from service.text.preprocessor import TextPreprocessor
from service.topic.aske import ASKETopicExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

graph = Neo4jGraph(config.NEO4J_URI, config.NEO4J_USERNAME, config.NEO4J_PASSWORD)

ASKE = ASKETopicExtractor(graph)
preprocessor = TextPreprocessor()

# Extract paragraphs from the KG and preprocess into sentence-level chunks
paragraphs = graph.get_paragraphs_from_kg()
chunks = preprocessor.to_chunks(paragraphs, skip_first=True)

logger.info("Total chunks extracted: %d", len(chunks))

# Run full ASKE cycle for N generations
N_GENERATIONS = 20      # Number of ASKE generations
ALPHA = 0.45            # Classification threshold
BETA = 0.4              # Terminology enrichment threshold
GAMMA = 10              # Max new terms per concept per generation

test_seeds = SEEDS_AI_DATA_FOCUSED_v2

logger.info(
    "Starting ASKE cycle — seeds: %d, alpha=%.2f, beta=%.2f, gamma=%d, generations=%d",
    len(test_seeds), ALPHA, BETA, GAMMA, N_GENERATIONS,
)

concepts, final_classifications = ASKE.run_aske_cycle(
    chunks=chunks,
    seeds=test_seeds,
    n_generations=N_GENERATIONS,
    alpha=ALPHA,
    beta=BETA,
    gamma=GAMMA,
)

active_concepts = [c for c in concepts if c.get("active", True)]
inactive_concepts = [c for c in concepts if not c.get("active", True)]

logger.info(
    "ASKE complete — Total concepts: %d, Active: %d, Inactive: %d",
    len(concepts), len(active_concepts), len(inactive_concepts),
)

# Aggregate topics by paragraph and update Neo4j
paragraph_topics = ASKE.aggregate_topics_by_paragraph(final_classifications, top_n=3)

updated_count = graph.update_paragraph_topics(paragraph_topics)
logger.info("Updated topics for %d paragraphs in Neo4j", updated_count)

# --- Final report ---
report_path = pathlib.Path("results/aske_report.txt")
report_path.parent.mkdir(parents=True, exist_ok=True)

with report_path.open("w", encoding="utf-8") as f:
    f.write("ASKE Topic Report\n")
    f.write("=" * 60 + "\n\n")

    for concept in sorted(active_concepts, key=lambda c: c["label"]):
        terms = concept.get("terms", [])
        term_labels = sorted({t["label"] if isinstance(t, dict) else t for t in terms})
        f.write(f"[{concept['label']}]\n")
        f.write(f"  Terms: {', '.join(term_labels)}\n\n")

logger.info("Report written to %s", report_path)
