
import config
import re
from typing import List, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pydantic import ConfigDict
from sentence_transformers import SentenceTransformer

from query import NodeQueries
from service.rag.prompt import ANSWER_SYNTHESIS_PROMPT

class GraphEnrichedRetriever(BaseRetriever):
    """Retriever combining semantic topic filtering with vector similarity search."""

    vector_store: Any
    graph: Any
    k: int = 5
    use_topic_filter: bool = True
    topic_top_k: int = 5
    topic_threshold: float = 0.35
    embedding_model_name: str = "all-MiniLM-L6-v2"

    _embedding_model: Any = None
    _topic_cache: dict = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def _load_topic_embeddings(self) -> dict:
        """Load and cache topic embeddings from the knowledge graph."""
        if self._topic_cache is not None:
            return self._topic_cache

        result = self.graph.query(NodeQueries.GET_ALL_UNIQUE_TOPICS)
        extracted_topics = [row["topic"] for row in result]

        if not extracted_topics:
            # return empty for consistency
            self._topic_cache = {"topics": [], "embeddings": np.array([])}
        else:
            embeddings = self.embedding_model.encode(extracted_topics, show_progress_bar=False)
            self._topic_cache = {"topics": extracted_topics, "embeddings": np.array(embeddings)}

        return self._topic_cache

    def _filter_and_rank_topics(self, topics: List[str], scores: np.ndarray) -> List[tuple]:
        """Filter topics by threshold and return top-k sorted by score descending."""
        matches = []

        for topic, score in zip(topics, scores):
            if score >= self.topic_threshold:
                matches.append((topic, float(score)))
        return matches

    def _match_topics(self, query: str) -> List[tuple]:
        """Find topics semantically similar to the query."""
        cache = self._load_topic_embeddings()

        if not cache["topics"]:
            return []

        query_embedding = self.embedding_model.encode(query, show_progress_bar=False)
        similarities = cosine_similarity([query_embedding], cache["embeddings"])[0]

        matches = self._filter_and_rank_topics(cache["topics"], similarities)
        matches.sort(key=lambda match: match[1], reverse=True)

        return matches[:self.topic_top_k]

    def _get_paragraphs_by_topics(self, topics: List[str]) -> List[Document]:
        """Retrieve paragraphs associated with the given topics."""
        if not topics:
            return []

        results = self.graph.query(
            NodeQueries.GET_ALL_PARAGRAPHS_BY_TOPIC,
            params={"topics": topics, "limit": self.k}
        )

        return [
            Document(
                page_content=f"\ntext: {r['text']}\nid: {r['id']}",
                metadata={
                    "id": r["id"],
                    "topics": r["topics"],
                    "article_title": r["article_title"],
                    "source": "semantic_topic_filter"
                }
            )
            for r in results
        ]

    @staticmethod
    def _extract_paragraph_id(content: str) -> str:
        """Extract paragraph ID from document content."""
        match = re.search(r'\nid:\s*(\S+)', content)
        return match.group(1) if match else None

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        seen_ids = set()
        docs = []

        # Filter the KG based on semantic topic matching
        if self.use_topic_filter:
            matched_topics = self._match_topics(query)

            if matched_topics:
                print("[Semantic Topic Filter] Matched topics:")
                for topic, score in matched_topics:
                    print(f"  - {topic}: {score:.3f}")

                topic_names = [t for t, _ in matched_topics]
                for doc in self._get_paragraphs_by_topics(topic_names):
                    paragraph_id = doc.metadata.get("id")
                    if paragraph_id and paragraph_id not in seen_ids:
                        seen_ids.add(paragraph_id)
                        docs.append(doc)

        # Vector similarity search
        for doc in self.vector_store.similarity_search(query, k=self.k):
            paragraph_id = self._extract_paragraph_id(doc.page_content)
            if paragraph_id and paragraph_id not in seen_ids:
                doc.metadata["id"] = paragraph_id
                doc.metadata["source"] = "vector_search"
                seen_ids.add(paragraph_id)
                docs.append(doc)

        print(f"[Retriever] Returning {len(docs)} documents")
        return docs[:self.k * 2]


QA_PROMPT = PromptTemplate(
    template=ANSWER_SYNTHESIS_PROMPT,
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
    embedding_node_property="textEmbeddingOpenAI"
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
output_file = "../../results/rag_results.txt"

# Run all queries and store the result
for i, query in enumerate(queries, 1):
    print(f"\n{'='*70}")
    print(f"Query {i}/{len(queries)}: {query[:50]}...")
    print("="*70)

    result = qa_chain.invoke({"query": query})

    print(f"\nResponse:\n{result['result'][:200]}...")
    sources = [doc.metadata.get('id') for doc in result['source_documents']]
    print(f"\nSources: {sources}")

    # Save results to file
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"Query {i}: {query}\n\n")
        f.write(f"RAG Response:\n{result['result']}\n\n")
        f.write(f"Sources: {sources}\n")
        f.write(f"Expert response:\n{expert_response[i-1]}\n")
        f.write("=" * 70 + "\n\n")

print(f"\n{'='*70}")
print(f"All {len(queries)} queries completed!")
print(f"Results saved to: {output_file}")