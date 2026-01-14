from service.graph.graph import Neo4jGraph
from service.graph.graph_loader import GraphLoader
import config
import os

graph = Neo4jGraph(config.NEO4J_URI, config.NEO4J_USERNAME, config.NEO4J_PASSWORD)

# Verify connection
graph.verify_connection()

#graph.clear_database()

# Initialize the loader
loader = GraphLoader(graph)

# Example: Load documents configuration
documents_config = [
    {
        'html_file': os.path.join(os.path.dirname(__file__), 'docs/GDPR.html'),
        'celex': '32016R0679',
        'author': 'European Parliament and Council',
        'publication_date': '27/04/2016',
        'date_of_application': '25/05/2018',
        'eurolex_url': 'https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679',
        'document_info_url': 'https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX%3A32016R0679'
    },
    {
        'html_file': os.path.join(os.path.dirname(__file__), 'docs/AI_ACT.html'),
        'celex': '32024R1689',
        'author': 'European Parliament and Council',
        'publication_date': '13/06/2024',
        'date_of_application': '01/08/2024',
        'eurolex_url': 'https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689',
        'document_info_url': 'https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX%32024R1689'
    },
    {
        'html_file': os.path.join(os.path.dirname(__file__), 'docs/Data Act.html'),
        'celex': '32023R2854',
        'author': 'European Parliament and Council',
        'publication_date': '13/12/2023',
        'date_of_application': '11/01/2024',
        'eurolex_url': 'https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32023R2854',
        'document_info_url': 'https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX%32023R2854'
    },
    {
        'html_file': os.path.join(os.path.dirname(__file__), 'docs/Data Governance Act.html'),
        'celex': '32022R0868',
        'author': 'European Parliament and Council',
        'publication_date': '30/05/2022',
        'date_of_application': '23/06/2022',
        'eurolex_url': 'https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32022R0868',
        'document_info_url': 'https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX%32022R0868'
    },
]

# Load all documents into the graph
loader.load_all_documents(documents_config)

# Close connection when done
graph.close()
print("All documents loaded successfully!")