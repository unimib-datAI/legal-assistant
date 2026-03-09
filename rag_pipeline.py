import logging
import pathlib

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config
from service.rag.prompt import ANSWER_SYNTHESIS_PROMPT_v2
from service.rag.rag_naive_with_topics import GraphEnrichedRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

QA_PROMPT = PromptTemplate(
    template=ANSWER_SYNTHESIS_PROMPT_v2,
    input_variables=["context", "question"]
)

graph = Neo4jGraph(
    url=config.NEO4J_URI,
    username=config.NEO4J_USERNAME,
    password=config.NEO4J_PASSWORD
)

# 1. Vector search for semantic similarity
vector_store = Neo4jVector.from_existing_graph(
    embedding=OpenAIEmbeddings(
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
    use_topic_filter=True  # Enable topic-based filtering
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT}
)

# Define multiple queries to run
queries = [
    "What does the data minimisation principle require in practice?",
    "What are the controller’s obligations in order to maintain data accuracy?",
    "How should the storage limitation principle and data retention schedules be applied?",
    "What does security of processing require under Article 32 of the GDPR?",
    "What is a Transfer Impact Assessment (TIA) and what should it contain?"
]

# Expert responses for comparison
expert_response = [
    "Data minimisation obliges controllers to ensure personal data are adequate, relevant and limited to what is necessary in relation to the stated purposes. Practically, this means designing data collection forms and systems to avoid optional or extra information that is not purpose-critical; using coarse or derived attributes instead of raw data when sufficient; and setting retention windows aligned to genuine operational or legal needs. For analytics and AI, privacy-preserving features should be preferred, with the use of synthetic or aggregated data where possible, and sampling should be used to reduce granularity of personal data. Purpose tags and minimisation checks should be built into engineering workflows (schema reviews, pull-request checklists). The organisation should map where each attribute is used; if unused, the attribute should be removed. With respect to vendors, organisations should ensure that contracts prohibit unnecessary collection or repurposing of personal data, and require deletion of personal data on termination. Minimisation also applies to access—role-based and time-bound permissions—and to the sharing of personal data with recipients. The rationale for the collection and retention of each category of data should be documented in the organisation’s records of processing activities and DPIAs, and should be revisited periodically as requirements evolve. Minimisation is not merely about collecting less; it is about collecting the right data—no more, no less—supported by evidence of necessity.",
    "Controllers must ensure personal data are accurate and, where necessary, kept up to date; every reasonable step must be taken to erase or rectify without delay data that are inaccurate for the purposes. Accuracy by design should be embedded: input validation; user selfservice to update details; periodic prompts to confirm core attributes; and reconciliation against authoritative sources where lawful. When accuracy is disputed, evidence should be assessed and, if appropriate, processing should be restricted while verification takes place. Factual inaccuracies should be distinguished from contested opinions; the latter may be rectified by adding context rather than overwriting. In data sharing, responsibility and feedback channels with recipients should be established to facilitate any necessary corrections of the personal data. For AI systems, provenance should be tracked and pipelines updated so training/feature stores can be corrected. Organisations should consider whether inaccuracies have produced downstream decisions that need reevaluation. Audit trails of rectification requests and actions taken should be maintained to demonstrate accountability. Accuracy interacts with fairness— the aim of the accuracy principle is to avoid negative impacts caused by stale or erroneous data.",
    "Storage limitation requires keeping personal data in identifiable form no longer than is necessary for the purposes. Documented retention schedules must be implemented, linked to purpose, legal obligations and limitation periods. Data lifecycle tooling should be employed to enable automatic deletion, archiving, and irreversible anonymisation where ongoing analysis is required without identification. Business-critical retention (e.g., tax records) must be distinguished from convenience storage, as convenience alone rarely justifies extended retention. Privacy notices must specify retention periods clearly, preferably through defined ranges or rules (e.g., “invoices are retained for 10 years per tax law; support chats for 24 months”). In shared datasets and data lakes, per-attribute retention and masking should be applied rather than uniform rules across the board. For backups, restoration windows and deletion procedures upon restore must be defined. In AI and analytics contexts, rolling windows and retraining workflows should be designed respect retention, with documentation of how models handle deletion requests (e.g., retraining, machine unlearning). Deletion controls must be periodically tested and evidence retained. Where law requires longer retention or preservation (e.g., litigation holds), access and use must be restricted solely to that purpose.",
    "Security must be appropriate to the risk, considering the state of the art, costs, nature, scope, context and purposes of processing, and risks to individuals. Measures may include pseudonymisation and encryption, maintenance of confidentiality, integrity, availability and resilience of systems, the ability to restore availability of and access to data in a timely manner, and processes for regularly testing, assessing and evaluating effectiveness. A risk assessment must be conducted in relation to processing operations and applicable threat model, with alignment to recognised security frameworks (e.g. ISO 27001, NIST) where beneficial. Least-privilege access, multi-factor authentication, network segmentation, secure software development, vulnerability management, logging and monitoring, and incident response plans must be implemented. Equivalent controls must be required of processors by contract (in terms of Article 28 of GDPR) and verified through audits. Data protection by design and default must complement security by minimising data, restricting access and masking by default. Security documentation and evidence (policies, penetration tests, disaster recovery tests) must be maintained. Security is not static – reviews must be documented following incidents, significant changes or emerging threats.",
    "A TIA evaluates whether, for a given transfer of personal data to a third country on the basis of the SCCs or other safeguards, the laws and practices in the destination country may impinge on the effectiveness of the transfer tool. The TIA should document: (1) the transfer details (parties, roles, data categories, purposes, recipients, onward transfers); (2) the transfer tool used (e.g., SCCs module); (3) an assessment of the third country’s legal landscape, particularly with respect to government access to data and redress; (4) the likelihood of problematic access to personal data in practice; (5) technical and organisational measures (e.g., strong encryption with key control in the EEA, split processing, transparency); and (6) conclusions and residual risks, including whether to proceed with the proposed transfer or adopt additional measures. Evidence and sources relied upon in the TIA should be maintained and the TIA should be re-assessed periodically or upon any material change which may impact the transfer. TIAs should be integrated with vendor due diligence and contract management."
]

# Output file
output_file = pathlib.Path("results/prova.txt")

# Run all queries and store the result
for i, query in enumerate(queries, 1):
    logger.info("Query %d/%d", i, len(queries))

    result = qa_chain.invoke({"query": query})

    answer = result['result'].replace('\r\n', '\n').replace('\r', '\n').strip()
    sources = [doc.metadata.get('id') for doc in result['source_documents']]

    with open(output_file, "a", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"Query {i}: {query}\n\n")
        f.write(f"RAG Response:\n{answer}\n\n")
        f.write(f"Sources: {sources}\n")
        f.write(f"Expert response:\n{expert_response[i-1]}\n")
        f.write("=" * 70 + "\n\n")

logger.info("All %d queries completed. Results saved to: %s", len(queries), output_file)