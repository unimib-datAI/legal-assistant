import json
import logging
import pathlib

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config
from service.rag.prompt import ANSWER_SYNTHESIS_PROMPT
from service.rag.rag_naive_with_topics import GraphEnrichedRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

QA_PROMPT = PromptTemplate(
    template=ANSWER_SYNTHESIS_PROMPT,
    input_variables=["context", "question"]
)

graph = Neo4jGraph(
    url=config.NEO4J_URI,
    username=config.NEO4J_USERNAME,
    password=config.NEO4J_PASSWORD
)

vector_store = Neo4jVector.from_existing_graph(
    embedding=OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=config.OPENAI_API_KEY
    ),
    url=config.NEO4J_URI,
    username=config.NEO4J_USERNAME,
    password=config.NEO4J_PASSWORD,
    index_name="Paragraph",
    node_label="Paragraph",
    text_node_properties=["text", "id"],
    embedding_node_property="textEmbedding"
)

retriever = GraphEnrichedRetriever(
    vector_store=vector_store,
    graph=graph,
    k=5,
    use_topic_filter=True
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT}
)

questions_file = pathlib.Path("compliance_questions.json")
questions = json.loads(questions_file.read_text(encoding="utf-8"))

output_file = pathlib.Path("results/rag_result.json")
output_file.parent.mkdir(parents=True, exist_ok=True)

report: list[dict] = []

for i, item in enumerate(questions, 1):
    query: str = item["query"]
    expert_response: str = item["expert_response"]

    logger.info("Query %d/%d: %s", i, len(questions), query)

    result = qa_chain.invoke({"query": query})

    rag_response: str = result["result"].replace("\r\n", "\n").replace("\r", "\n").strip()
    sources: list[str] = [doc.metadata.get("id") for doc in result["source_documents"]]

    report.append({
        "query": query,
        "rag_response": rag_response,
        "expert_response": expert_response,
        "sources": sources,
    })

output_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
logger.info("All %d queries completed. Results saved to: %s", len(questions), output_file)