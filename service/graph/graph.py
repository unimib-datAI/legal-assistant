import logging
import sys
from pathlib import Path
from typing import Callable

from neo4j import GraphDatabase
from tqdm import tqdm

# Add root directory to path to import query module
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from service.graph.query import GeneralQueries, NodeQueries, RelationQueries

logger = logging.getLogger(__name__)

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
        logger.info("Connected to Neo4j successfully!")

    def clear_database(self):
        """Clear all nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")

    def create_graph_node(self, node_name, node_properties):
        """Create a node with the given name and properties, returning its ID."""
        with self.driver.session() as session:
            query = NodeQueries.CREATE_NODE.format(node_name=node_name)
            result = session.run(query, node_properties=node_properties)
            node_id = result.single()["node_id"]
            logger.info("Created %s node (ID: %s)", node_name, node_id)
            return node_id

    def create_relationship(self, left_node_name, right_node_name, left_id, right_id, relationship):
        """Create a relationship of the specified type between two nodes identified by their IDs."""
        with self.driver.session() as session:
            query = RelationQueries.CREATE_RELATIONSHIP.format(
                left_node_name=left_node_name,
                right_node_name=right_node_name,
                relationship=relationship
            )
            session.run(query, left_id=left_id, right_id=right_id)
            logger.info("Created %s relationship between %s(id=%s) and %s(id=%s)",
                        relationship, left_node_name, left_id, right_node_name, right_id)

    def create_vector_index(self, node_name, index_name, dimensions: int):
        """Create a vector index on the specified node label and property."""
        with self.driver.session() as session:
            query = GeneralQueries.CREATE_VECTOR_INDEX.format(
                node_name=node_name,
                index_name=index_name,
                dimensions=dimensions,
            )
            session.run(query)
            logger.info("Created vector index %s on %s nodes (dimensions=%d)", index_name, node_name, dimensions)

    def generate_text_embeddings(
        self,
        embed_fn: Callable[[list[str]], list[list[float]]],
        embedding_dim: int,
        node_name: str,
        batch_size: int = 32,
    ) -> int:
        """Generate embeddings for nodes missing them using any embedding callable.

        Args:
            embed_fn: Callable that takes a list of strings and returns a list of float vectors.
            embedding_dim: Dimensionality of the vectors produced by embed_fn.
            node_name: The Neo4j node label to embed (e.g. "Paragraph").
            batch_size: Number of texts to encode at once.

        Returns:
            The embedding dimension.
        """
        with self.driver.session() as session:
            retrieve_query = NodeQueries.GET_NODE_WITHOUT_EMBEDDING.format(node_name=node_name)
            nodes = list(session.run(retrieve_query).data())
            logger.info("Found %d %s nodes without embeddings", len(nodes), node_name)

            if not nodes:
                return embedding_dim

            with tqdm(total=len(nodes), desc=f"Embedding {node_name} nodes") as pbar:
                for i in range(0, len(nodes), batch_size):
                    batch = nodes[i:i + batch_size]
                    texts = [record['text'] for record in batch]
                    embeddings = embed_fn(texts)

                    for record, vector in zip(batch, embeddings):
                        update_query = NodeQueries.PUT_EMBEDDING.format(node_name=node_name)
                        session.run(update_query, node_id=record['node_id'], vector=vector)
                        pbar.update(1)

        return embedding_dim

    def get_paragraphs_from_kg(self):
        """Extract paragraphs from Knowledge Graph"""
        paragraphs = []

        with self.driver.session() as session:
            query = NodeQueries.GET_ALL_PARAGRAPHS.format()
            result = session.run(query)

            for record in result:
                paragraphs.append({
                    "paragraph_id": record["paragraph_id"],
                    "text": record["paragraph_text"]
                })

        logger.info("Extracted %d paragraphs", len(paragraphs))
        return paragraphs

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

        with self.driver.session() as session:
            for paragraph_id, topics in paragraph_topics.items():
                for topic_dict in topics:
                    topic_label = topic_dict["topic"]

                    if topic_label not in created_topics:
                        session.run(
                            NodeQueries.CREATE_TOPIC_NODE,
                            topic_label=topic_label,
                        )
                        created_topics.add(topic_label)

                    session.run(
                        NodeQueries.CREATE_PARAGRAPH_TOPIC_RELATIONSHIP,
                        paragraph_id=paragraph_id,
                        topic_label=topic_label,
                    )

                updated_count += 1

        logger.info("Created %d Topic nodes", len(created_topics))
        logger.info("Linked %d paragraphs to topics via RELATED_TO", updated_count)
        return updated_count
