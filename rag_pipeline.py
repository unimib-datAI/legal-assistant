import json
import logging
import pathlib

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

import config
from service.rag.intent_classifier import QueryClassifier
from service.rag.prompt import ANSWER_SYNTHESIS_PROMPT, ANSWER_FILTER_PROMPT
from service.rag.rag_naive_with_topics import GraphEnrichedRetriever
from service.rag.rag_alternative import ArticleTraversalRetriever

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


class RAGPipeline:

    def __init__(self, use_answer_filter: bool = False):
        self.use_answer_filter = use_answer_filter
        graph = Neo4jGraph(
            url=config.NEO4J_URI,
            username=config.NEO4J_USERNAME,
            password=config.NEO4J_PASSWORD
        )

        vector_store = Neo4jVector.from_existing_graph(
            embedding=HuggingFaceEmbeddings(
                model_name="BAAI/bge-large-en-v1.5",
                encode_kwargs={"normalize_embeddings": True},
            ),
            url=config.NEO4J_URI,
            username=config.NEO4J_USERNAME,
            password=config.NEO4J_PASSWORD,
            index_name="Paragraph",
            node_label="Paragraph",
            text_node_properties=["text", "id"],
            embedding_node_property="textEmbedding"
        )

        classifier_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
        )
        classifier = QueryClassifier(graph=graph, llm=classifier_llm)

        # retriever = GraphEnrichedRetriever(
        #     vector_store=vector_store,
        #     graph=graph,
        #     k=15,
        #     use_topic_filter=True,
        #     top_k_topic_paragraphs=20,
        #     filter_by_act=False,
        #     classifier=classifier,
        # )

        retriever = ArticleTraversalRetriever(
            graph=graph,
            classifier=classifier,
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0, api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )

        self.filter_llm = (
            ChatOpenAI(
                temperature=0,
                api_key=config.OPENAI_API_KEY,
                base_url=config.OPENAI_BASE_URL,
            )
            if self.use_answer_filter
            else None
        )

    def query(self, question: str) -> dict:
        result = self.qa_chain.invoke({"query": question})
        answer = result["result"].replace("\r\n", "\n").replace("\r", "\n").strip()

        if self.use_answer_filter:
            filtered = self.filter_llm.invoke(
                ANSWER_FILTER_PROMPT.format(question=question, draft_answer=answer)
            )
            answer = filtered.content.replace("\r\n", "\n").replace("\r", "\n").strip()

        return {
            "answer": answer,
            "sources": [doc.metadata.get("id") for doc in result["source_documents"]],
            "contexts": [doc.page_content for doc in result["source_documents"]],
        }

    def run_batch(self, questions_path: str, output_path: str) -> None:
        questions = json.loads(pathlib.Path(questions_path).read_text(encoding="utf-8"))
        output_file = pathlib.Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        report = []
        for i, item in enumerate(questions, 1):
            logger.info("Query %d/%d: %s", i, len(questions), item["question"])
            response = self.query(item["question"])
            report.append({
                "act": item["act"],
                "question": item["question"],
                "rag_response": response["answer"],
                "ground_truth": item["ground_truth"],
                "retrieved_context": response["sources"],
            })

        output_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Done. Results saved to: %s", output_file)



if __name__ == "__main__":
    rag = RAGPipeline()
    rag.run_batch("docs/question_recital_required.json", "results/rag_result.json")