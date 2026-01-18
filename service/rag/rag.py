import os
import sys
from pathlib import Path

# Add the root directory to the Python path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

import re
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import config

graph = Neo4jGraph(
    url=config.NEO4J_URI,
    username=config.NEO4J_USERNAME,
    password=config.NEO4J_PASSWORD
)

os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY

# Create vector index for Article and Paragraph nodes
article_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    url=config.NEO4J_URI,
    username=config.NEO4J_USERNAME,
    password=config.NEO4J_PASSWORD,
    index_name='Article',
    node_label="Article",
    text_node_properties=['title', 'full_text'],
    embedding_node_property='textEmbeddingOpenAI',
)

paragraph_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    url=config.NEO4J_URI,
    username=config.NEO4J_USERNAME,
    password=config.NEO4J_PASSWORD,
    index_name='Paragraph',
    node_label="Paragraph",
    text_node_properties=['text'],
    embedding_node_property='textEmbeddingOpenAI',
)

# Custom retriever function that combines both indexes
def combined_retriever(query):
    article_results = article_index.similarity_search(query, k=5)
    paragraph_results = paragraph_index.similarity_search(query, k=5)
    return article_results + paragraph_results

# Create a RAG chain using the modern LangChain approach
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    print("=== RETRIEVED DOCUMENTS ===")
    for i, doc in enumerate(docs):
        print(f"\nDoc {i+1}:")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")
    print("=" * 50)
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": lambda x: format_docs(combined_retriever(x)),
        "question": RunnablePassthrough()
    }
    | prompt
    | ChatOpenAI()
    | StrOutputParser()
)

result = rag_chain.invoke("Summarize the second paragraph (2.) of the 18 article of GDPR.")
print(result)