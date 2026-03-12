import logging

import config
from langchain_openai import OpenAIEmbeddings
from service.graph.graph import Neo4jGraph
from service.graph.graph_loader import GraphLoader
from service.scraper.eurlex_document_utils import EurlexDocumentUtils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

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

# Generate embeddings for Paragraph nodes using OpenAI text-embedding-3-small
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=config.OPENAI_API_KEY)
dimensions = graph.generate_text_embeddings(
    embed_fn=openai_embeddings.embed_documents,
    embedding_dim=1536,
    node_name="Paragraph",
)

# Create vector index for similarity search
graph.create_vector_index("Paragraph", "Paragraph", dimensions)

# Close connection when done
graph.close()
logger.info("All documents loaded successfully!")