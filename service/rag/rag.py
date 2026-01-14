import os
import sys
from pathlib import Path

# Add the root directory to the Python path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import config

graph = Neo4jGraph(
    url=config.NEO4J_URI,
    username=config.NEO4J_USERNAME,
    password=config.NEO4J_PASSWORD
)

os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY

# Create vector index for Article nodes
vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    url=config.NEO4J_URI,
    username=config.NEO4J_USERNAME,
    password=config.NEO4J_PASSWORD,
    index_name='articles',
    node_label="Article",
    text_node_properties=['title', 'full_text'],
    embedding_node_property='embedding',
)

# Search for relevant articles
respond = vector_index.similarity_search(
    "What says the article 2 of AI Act?"
)

# Create a RAG chain using the modern LangChain approach
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": vector_index.as_retriever() | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI()
    | StrOutputParser()
)

result = rag_chain.invoke("What says the article 2 of AI Act?")
print(result)