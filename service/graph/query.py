"""
Neo4j Cypher query repository
"""

# rel: https://community.neo4j.com/t/creating-a-relationship-between-2-nodes/38408/2
# rel optimal way: https://stackoverflow.com/questions/69702238/neo4j-cypher-iteratively-creating-relationships-in-optimal-way
# https://www.quackit.com/neo4j/tutorial/neo4j_create_a_relationship_using_cypher.cfm


class NodeQueries:
    """Queries for node operations"""

    EXISTS_NODE = """
    MATCH (n:{node_name} {{id: $node_id}})
    RETURN count(n) > 0 AS exists
    """

    CREATE_NODE = """
    MERGE (n:{node_name} {{id: $node_properties.id}})
    SET n = $node_properties
    RETURN n.id as node_id
    """
    GET_NODE_WITHOUT_EMBEDDING = """
    MATCH (n:{node_name})
    WHERE n.textEmbedding IS NULL
    return n.id as node_id, n.text as text
    """
    PUT_EMBEDDING = """
    MATCH (n:{node_name})
    WHERE n.id = $node_id
    CALL db.create.setNodeVectorProperty(n, "textEmbedding", $vector)
    """
    
    GET_ALL_PARAGRAPHS = """
    MATCH (act:Act)-[:CONTAINS*]->(art:Article)-[:CONTAINS]->(p:Paragraph)
    RETURN p.id as paragraph_id, p.text AS paragraph_text
    ORDER BY art.id, p.id
    """

    CREATE_TOPIC_NODE = """
    MERGE (t:Topic {label: $topic_label})
    RETURN t.label as topic_label
    """

    CREATE_PARAGRAPH_TOPIC_RELATIONSHIP = """
    MATCH (p:Paragraph {id: $paragraph_id})
    MATCH (t:Topic {label: $topic_label})
    MERGE (p)-[:RELATED_TO]->(t)
    RETURN p.id as paragraph_id, t.label as topic_label
    """

    GET_ALL_UNIQUE_TOPICS = """
    MATCH (t:Topic)
    RETURN collect(t.label) AS topics
    """

    GET_ALL_PARAGRAPHS_BY_TOPIC = """
    MATCH (p:Paragraph)-[:RELATED_TO]->(t:Topic)
    WHERE t.label IN $topics
    MATCH (p)<-[:CONTAINS]-(a:Article)
    MATCH (p)<-[:CONTAINS*]-(act:Act)
    WHERE size($acts) = 0 OR act.id IN $acts
    WITH p, a, collect(t.label) AS topics
    RETURN p.id AS id,
           p.text AS text,
           topics,
           a.title AS article_title
    """

    GET_ALL_RECITALS_BY_ACT = """
    MATCH (a:Act {id: $celex})-[:CONTAINS]->(r:Recital)
    RETURN r
    ORDER BY toInteger(r.number)
    """

    GET_ALL_ACTS = """
    MATCH (a:Act)
    RETURN a.id AS celex, a.title AS title
    ORDER BY a.id
    """

    GET_ALL_CHAPTERS_BY_ACTS = """
    MATCH (act:Act)-[:CONTAINS]->(c:Chapter)
    WHERE act.id IN $acts
    RETURN c.number   AS chapter_number,
           c.title    AS chapter_title,
           c.summary  AS chapter_summary,
           act.id     AS act_id
    ORDER BY act.id, c.number
    """

    GET_PARAGRAPHS_BY_CHAPTERS = """
    MATCH (act:Act)-[:CONTAINS]->(c:Chapter)
    WHERE act.id IN $acts AND c.number IN $chapter_numbers
    MATCH (c)-[:CONTAINS*1..2]->(art:Article)-[:CONTAINS]->(p:Paragraph)
    WITH p, art, toInteger(split(art.id, 'art_')[-1]) AS art_num
    RETURN p.id AS id, p.text AS text, art.title AS article_title, art.id AS article_id
    ORDER BY art_num, p.id
    """

    GET_ALL_CHAPTERS_WITHOUT_SUMMARY = """
    MATCH (act:Act)-[:CONTAINS]->(c:Chapter)
    WHERE c.summary IS NULL
    MATCH (c)-[:CONTAINS*1..2]->(art:Article)
    WITH act, c, collect(art.title) AS article_titles
    RETURN c.number     AS chapter_number,
           c.title      AS chapter_title,
           act.id       AS celex,
           act.title    AS act_title,
           article_titles
    ORDER BY act.id, c.number
    """

    UPDATE_CHAPTER_SUMMARY = """
    MATCH (act:Act {id: $celex})-[:CONTAINS]->(c:Chapter {number: $chapter_number})
    SET c.summary = $summary
    """

    GET_ALL_ARTICLES_WITH_PARAGRAPHS = """
    MATCH (act:Act)-[:CONTAINS*]->(art:Article)
    WHERE art.summary IS NULL
    WITH act, art
    MATCH (art)-[:CONTAINS]->(p:Paragraph)
    WITH act, art, collect(p.text) AS paragraphs
    RETURN art.id        AS article_id,
           art.title     AS article_title,
           act.id        AS celex,
           act.title     AS act_title,
           reduce(body = '', t IN paragraphs | body + '\n\n' + t) AS body
    ORDER BY act.id, art.id
    """

    UPDATE_ARTICLE_SUMMARY = """
    MATCH (art:Article {id: $article_id})
    SET art.summary = $summary
    """

    GET_ALL_ARTICLE_TITLES = """
    MATCH (act:Act)-[:CONTAINS*]->(art:Article)
    RETURN art.id AS article_id, art.title AS article_title, act.id AS act_id
    ORDER BY act.id, art.id
    """

    GET_ARTICLE_TITLES_BY_ACTS = """
    MATCH (act:Act)-[:CONTAINS*]->(art:Article)
    WHERE act.id IN $acts
    RETURN art.id AS article_id,
           art.title AS article_title,
           art.summary AS article_summary
    ORDER BY art.id
    """

    GET_PARAGRAPHS_BY_ARTICLES = """
    MATCH (art:Article)-[:CONTAINS]->(p:Paragraph)
    WHERE art.id IN $article_ids
    RETURN p.id AS id, p.text AS text, art.title AS article_title, art.id AS article_id
    ORDER BY art.id, p.id
    """

    GET_RECITALS_BY_ACTS = """
    MATCH (a:Act)-[:CONTAINS]->(r:Recital)
    WHERE a.id IN $acts
    RETURN r.id AS recital_id, r.text AS text, a.id AS celex
    ORDER BY a.id, toInteger(r.number)
    """

    GET_ARTICLES_BY_ACTS = """
    MATCH (act:Act)-[:CONTAINS*]->(art:Article)
    WHERE act.id IN $acts
    OPTIONAL MATCH (act)-[:CONTAINS]->(ch:Chapter)-[:CONTAINS*]->(art)
    RETURN art.id AS id, art.title AS title, art.text AS text, act.id AS act,
           ch.number AS chapter_number, ch.title AS chapter_title
    ORDER BY act.id, art.id
    """

    GET_GENERAL_PROVISIONS_BY_ACTS = """
    MATCH (act:Act)-[:CONTAINS*]->(art:Article)
    WHERE act.id IN $acts
    WITH act, art, toInteger(split(art.id, 'art_')[-1]) AS art_num
    WHERE art_num <= $max_article_num
    MATCH (art)-[:CONTAINS]->(p:Paragraph)
    RETURN p.id AS id, p.text AS text, art.title AS article_title, art.id AS article_id
    ORDER BY act.id, art_num, p.id
    """

    GET_CHAPTER_PARAGRAPHS = """
    MATCH (act:Act)-[:CONTAINS]->(c:Chapter)
    WHERE act.id IN $acts AND c.number = $chapter_number
    MATCH (c)-[:CONTAINS*1..2]->(art:Article)-[:CONTAINS]->(p:Paragraph)
    WITH p, art, toInteger(split(art.id, 'art_')[-1]) AS art_num
    RETURN p.id AS id, p.text AS text, art.title AS article_title, art.id AS article_id
    ORDER BY art_num, p.id
    """

    GET_DEFINITIONS_BY_ACTS = """
    MATCH (act:Act)-[:CONTAINS*]->(art:Article)
    WHERE act.id IN $acts AND art.title = 'Definitions'
    MATCH (art)-[:CONTAINS]->(p:Paragraph)
    RETURN p.id AS id, p.text AS text, art.title AS article_title, art.id AS article_id
    ORDER BY art.id, p.id
    """

class RelationQueries:
    """Queries for relation operations"""

    CREATE_RELATIONSHIP = """
    MATCH (ln:{left_node_name} {{id: $left_id}})
    MATCH (rn:{right_node_name} {{id: $right_id}})
    CREATE (ln)-[:{relationship}]->(rn)
    RETURN ln.id as left_id, rn.id as right_id
    """
    
class GeneralQueries:
    """General purpose queries"""
    
    DROP_INDEX_IF_EXISTS = "DROP INDEX {index_name} IF EXISTS"

    CREATE_VECTOR_INDEX = """
    CREATE VECTOR INDEX {index_name}
    FOR (n:{node_name}) ON (n.textEmbedding)
    OPTIONS {{ indexConfig: {{
        `vector.dimensions`: {dimensions},
        `vector.similarity_function`: 'COSINE'
        }}}}
    """