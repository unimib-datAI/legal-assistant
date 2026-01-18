from langchain_core.prompts import PromptTemplate
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_openai import ChatOpenAI
import config

graph = Neo4jGraph(
    url=config.NEO4J_URI,
    username=config.NEO4J_USERNAME,
    password=config.NEO4J_PASSWORD
)
graph.embed_text_openai(OPENAI_API_KEY=config.OPENAI_API_KEY, OPENAI_ENDPOINT=config.OPENAI_ENDPOINT, node_name="Article")
graph.embed_text_openai(OPENAI_API_KEY=config.OPENAI_API_KEY, OPENAI_ENDPOINT=config.OPENAI_ENDPOINT, node_name="Paragraph")
graph.create_vector_index("Article", "Article")
graph.create_vector_index("Paragraph", "Paragraph")


GRAPH_RAG_PROMPT_TEMPLATE = """You are an expert legal assistant specialized in EU law.

=== GRAPH SCHEMA ===

Node Types:
- (Act) - Legal acts/regulations with properties: id (CELEX), title, author, publication_date, date_of_application, eurolex_url
- (Chapter) - Document chapters with properties: id, number, title
- (Section) - Chapter sections with properties: id, title
- (Article) - Legal articles with properties: id, title, text
- (Paragraph) - Article paragraphs with properties: id, text
- (Recital) - Preamble recitals with properties: id, number, text
- (CaseLaw) - Court case references with properties: id (case identifier)

Relationships:
- (Act)-[:CONTAINS]->(Chapter) - Act contains chapters
- (Act)-[:CONTAINS]->(Recital) - Act contains recitals
- (Chapter)-[:CONTAINS]->(Section) - Chapter contains sections
- (Chapter)-[:CONTAINS]->(Article) - Chapter directly contains articles
- (Section)-[:CONTAINS]->(Article) - Section contains articles
- (Article)-[:CONTAINS]->(Paragraph) - Article contains paragraphs
- (Article)-[:CITES]->(Article) - Cross-references between articles
- (CaseLaw)-[:INTERPRETS]->(Article|Paragraph|Chapter) - Case law interpretations

=== QUERY GUIDELINES ===

1. Do not use title articles to query the article number; use text property instead.
2. Use OPTIONAL MATCH for related data that may not exist
3. Limit results appropriately for large result sets
4. IMPORTANT: Always assign variables to nodes in MATCH patterns. Never use patterns like (:Node).property - always use (n:Node) and then n.property

=== CURRENT SCHEMA ===
{schema}

=== USER QUESTION ===
{question}

"""

def generate_cypher_query(question: str, graph):
    cypher_prompt = PromptTemplate(
        input_variables=["question", "schema"],
        template=GRAPH_RAG_PROMPT_TEMPLATE
    )
    
    cypher_chain = GraphCypherQAChain.from_llm(
        ChatOpenAI(temperature=0, api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL or None),
        graph=graph,
        verbose=True,
        prompt=cypher_prompt,
        allow_dangerous_requests=True
    )
    
    response = cypher_chain.run(question)

    return response

result = generate_cypher_query(question="Summarize the Article 2 of GDPR. Article number is usually contained in the text, not the title.", graph=graph)
#result = generate_cypher_query(question="Summarize the first paragraph of Article 2 of GDPR. Article number is usually contained in the text, not the title.", graph=graph)
print(result)