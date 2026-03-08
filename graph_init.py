import torch
from sentence_transformers import SentenceTransformer

from service.graph.graph import Neo4jGraph
from service.graph.graph_loader import GraphLoader
from service.scraper.eurlex_document_utils import EurlexDocumentUtils
import config

graph = Neo4jGraph(config.NEO4J_URI, config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
eurlex_document_utils = EurlexDocumentUtils()

# Verify connection
graph.verify_connection()

# OPTIONAL: Clear existing data if NEEDED, if not comment out the line below
graph.clear_database()

# Initialize the loader
loader = GraphLoader(graph)

# CELEX to download
celex_ids = ["32016R0679", "32024R1689", "32023R2854", "32022R0868"]

# Build document configurations
documents_config = [eurlex_document_utils.build_document_config(celex) for celex in celex_ids]

# Load all documents into the graph
loader.load_all_documents(documents_config)

# Generate Qwen embeddings for Paragraph nodes
embedding_model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B",
    model_kwargs={"dtype": torch.float16}
)
dimensions = graph.embed_text(embedding_model, "Paragraph", batch_size=4)

# Create vector index for similarity search
graph.create_vector_index("Paragraph", "Paragraph", dimensions)

# Close connection when done
graph.close()
print("All documents loaded successfully!")