from neo4j import GraphDatabase
from pathlib import Path
import sys
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Add root directory to path to import query module
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from service.graph.query import GeneralQueries, NodeQueries, RelationQueries
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
        """Create a node with the given name and properties, returning its ID."""
        with self.driver.session() as session:
            query = NodeQueries.CREATE_NODE.format(node_name=node_name)
            result = session.run(query, node_properties=node_properties)
            node_id = result.single()["node_id"]
            log.info(f"Created {node_name} node (ID: {node_id})")
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
            log.info(f"Created {relationship} relationship between {left_node_name}(id={left_id}) and {right_node_name}(id={right_id})")
            
    def create_vector_index(self, node_name, index_name, dimensions: int):
        """Create a vector index on the specified node label and property."""
        with self.driver.session() as session:
            query = GeneralQueries.CREATE_VECTOR_INDEX.format(
                node_name=node_name,
                index_name=index_name,
                dimensions=dimensions,
            )
            session.run(query)
            log.info(f"Created vector index {index_name} on {node_name} nodes (dimensions={dimensions})")

    def generate_text_embeddings(self, model: SentenceTransformer, node_name: str, batch_size: int = 32) -> int:
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

        print(f"Extracted {len(paragraphs)} paragraphs")
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
