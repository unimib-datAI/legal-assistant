from neo4j import GraphDatabase
from pathlib import Path
import sys
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Add root directory to path to import query module
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from query import GeneralQueries, NodeQueries, RelationQueries
import logging as log

log.basicConfig(level=log.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

"""Neo4j Graph creation utilities."""

class Neo4jGraph:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        if self.driver is not None:
            self.driver.close()

    def verify_connection(self):
        """Test database connection"""
        self.driver.verify_connectivity()
        log.info("Connected to Neo4j successfully!")

    def clear_database(self):
        """Clear all nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            log.info("Database cleared")

    def create_graph_node(self, node_name, node_properties):
        with self.driver.session() as session:
            query = NodeQueries.CREATE_NODE.format(node_name=node_name)
            result = session.run(query, node_properties=node_properties)
            node_id = result.single()["node_id"]
            log.info(f"Created {node_name} node (ID: {node_id})")
            return node_id
        
    def create_relationship(self, left_node_name, right_node_name, left_id, right_id, relationship):
        with self.driver.session() as session:
            query = RelationQueries.CREATE_RELATIONSHIP.format(
                left_node_name=left_node_name,
                right_node_name=right_node_name,
                relationship=relationship
            )
            session.run(query, left_id=left_id, right_id=right_id)
            log.info(f"Created {relationship} relationship between {left_node_name}(id={left_id}) and {right_node_name}(id={right_id})")
            
    def create_vector_index(self, node_name, index_name, dimensions: int):
        with self.driver.session() as session:
            query = GeneralQueries.CREATE_VECTOR_INDEX.format(
                node_name=node_name,
                index_name=index_name,
                dimensions=dimensions,
            )
            session.run(query)
            log.info(f"Created vector index {index_name} on {node_name} nodes (dimensions={dimensions})")

    def embed_text(self, model: SentenceTransformer, node_name: str, batch_size: int = 32) -> int:
        """Generate embeddings for nodes missing them using a SentenceTransformer model.

        Args:
            model: A loaded SentenceTransformer model.
            node_name: The Neo4j node label to embed (e.g. "Paragraph").
            batch_size: Number of texts to encode at once.

        Returns:
            The embedding dimension produced by the model.
        """
        with self.driver.session() as session:
            retrieve_query = NodeQueries.GET_NODE_WITHOUT_EMBEDDING.format(node_name=node_name)
            nodes = list(session.run(retrieve_query).data())
            log.info(f"Found {len(nodes)} {node_name} nodes without embeddings")

            if not nodes:
                # Return dimension from a dummy encode so callers can still create the index
                return model.get_sentence_embedding_dimension()

            with tqdm(total=len(nodes), desc=f"Embedding {node_name} nodes") as pbar:
                for i in range(0, len(nodes), batch_size):
                    batch = nodes[i:i + batch_size]
                    texts = [record['text'] for record in batch]
                    embeddings = model.encode(texts, show_progress_bar=False)

                    for record, vector in zip(batch, embeddings):
                        update_query = NodeQueries.PUT_EMBEDDING.format(node_name=node_name)
                        session.run(update_query, node_id=record['node_id'], vector=vector.tolist())
                        pbar.update(1)

        return model.get_sentence_embedding_dimension()