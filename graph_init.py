from sentence_transformers import SentenceTransformer

import config
from service.graph.graph import Neo4jGraph
from service.graph.graph_loader import GraphLoader
from service.scraper.eurlex_document_utils import EurlexDocumentUtils

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
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimensions = graph.generate_text_embeddings(embedding_model, "Paragraph", batch_size=4)

# Create vector index for similarity search
graph.create_vector_index("Paragraph", "Paragraph", dimensions)

# Close connection when done
graph.close()
print("All documents loaded successfully!")