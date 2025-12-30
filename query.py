"""
Neo4j Cypher query repository
"""

# rel: https://community.neo4j.com/t/creating-a-relationship-between-2-nodes/38408/2
# rel optimal way: https://stackoverflow.com/questions/69702238/neo4j-cypher-iteratively-creating-relationships-in-optimal-way
# https://www.quackit.com/neo4j/tutorial/neo4j_create_a_relationship_using_cypher.cfm


class NodeQueries:
    """Queries for node operations"""

    CREATE_NODE = """
    MERGE (n:{node_name} {{id: $node_properties.id}})
    SET n = $node_properties
    RETURN n.id as node_id
    """

class RelationQueries:
    """Queries for relation operations"""

    CREATE_RELATIONSHIP = """
    MATCH (ln:{left_node_name}), (rn:{right_node_name})
    WHERE ln.id = $left_id AND rn.id = $right_id
    CREATE (ln)-[:{relationship}]->(rn)
    RETURN ln.id as left_id, rn.id as right_id

    """