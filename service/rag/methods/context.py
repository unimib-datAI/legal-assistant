"""Shared, expensive RAG resources built once and reused across methods.

Constructing the Neo4j connection, the vector stores (which use the OpenAI
`text-embedding-3-small` embedding model), the cross-encoders, and the LLM clients is costly, so a single
:class:`RagContext` is built once (cached by the frontend) and handed to every
:class:`~service.rag.methods.base.RagMethod`.
"""
from __future__ import annotations

import logging
from functools import cached_property

from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config
from service.rag.intent_classifier import QueryClassifier
from service.rag.rag_alternative import HyDEGenerator

logger = logging.getLogger(__name__)


class RagContext:
    """Holds the shared graph, vector stores, classifier and LLM clients."""

    def __init__(self) -> None:
        logger.info("[RagContext] Building shared resources…")
        self.graph = Neo4jGraph(
            url=config.NEO4J_URI,
            username=config.NEO4J_USERNAME,
            password=config.NEO4J_PASSWORD,
        )

        # One embedding instance shared by both vector stores. Must match the model used at
        # graph-build time (see frontend/kg/graph_init.py); both read config.EMBEDDING_MODEL.
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
            # Empty string would be passed to the OpenAI client verbatim (it only falls back on
            # None, not falsy) and break the URL, so normalize "" → None to hit the default endpoint.
            base_url=config.OPENAI_BASE_URL or None,
        )

        self.article_vector_store = Neo4jVector.from_existing_graph(
            embedding=self.embeddings,
            url=config.NEO4J_URI,
            username=config.NEO4J_USERNAME,
            password=config.NEO4J_PASSWORD,
            index_name="Article",
            node_label="Article",
            text_node_properties=["text", "id", "title"],
            embedding_node_property="textEmbedding",
        )

        classifier_llm = ChatOpenAI(
            model=config.RAG_LLM_MODEL,
            temperature=0,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
        )
        self.classifier = QueryClassifier(graph=self.graph, llm=classifier_llm)

        self.synthesis_llm = ChatOpenAI(
            model=config.RAG_LLM_MODEL,
            temperature=0,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
        )

        self.filter_llm = ChatOpenAI(
            model=config.RAG_LLM_MODEL,
            temperature=0,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
        )

        # Cheap LLM for the optional pre-synthesis context-curation stage.
        self.curator_llm = ChatOpenAI(
            model=config.RAG_LLM_MODEL,
            temperature=0,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
        )
        logger.info("[RagContext] Ready.")

    @cached_property
    def paragraph_vector_store(self) -> Neo4jVector:
        """Paragraph-level store required by the topics retriever.

        Built lazily on first use so methods that don't need it (e.g. the hybrid
        retriever and the batch pipeline) don't pay its construction cost. The
        Paragraph vector index is created during graph initialization.
        """
        logger.info("[RagContext] Building paragraph vector store…")
        return Neo4jVector.from_existing_graph(
            embedding=self.embeddings,
            url=config.NEO4J_URI,
            username=config.NEO4J_USERNAME,
            password=config.NEO4J_PASSWORD,
            index_name="Paragraph",
            node_label="Paragraph",
            text_node_properties=["text", "id"],
            embedding_node_property="textEmbedding",
        )

    @cached_property
    def case_law_vector_store(self) -> Neo4jVector:
        """Judgment-paragraph store, used only on the INTERPRETIVE branch.

        Built lazily: a corpus with no case law ingested (see ``case_law_init.py``) has no
        CaseLawParagraph vector index, and DEFINITIONAL queries never touch this store.
        """
        logger.info("[RagContext] Building case law vector store…")
        return Neo4jVector.from_existing_graph(
            embedding=self.embeddings,
            url=config.NEO4J_URI,
            username=config.NEO4J_USERNAME,
            password=config.NEO4J_PASSWORD,
            index_name="CaseLawParagraph",
            node_label="CaseLawParagraph",
            text_node_properties=["text", "id"],
            embedding_node_property="textEmbedding",
        )

    def make_hyde_generator(self, iterations: int) -> HyDEGenerator:
        """Build a HyDE generator; temperature > 0 only when sampling >1 doc so
        the hypothetical passages diverge."""
        hyde_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7 if iterations > 1 else 0,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
        )
        return HyDEGenerator(llm=hyde_llm, iterations=iterations)
