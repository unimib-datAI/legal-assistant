"""Shared, expensive RAG resources built once and reused across methods.

Constructing the Neo4j connection, the vector stores, the cross-encoders and the LLM
clients is costly, so a single :class:`RagContext` is built once and handed to every
:class:`~legal_assistant.rag.methods.base.RagMethod`. A method reaches for what it needs
and pays nothing for the rest: the two node-level stores are lazy.
"""
from __future__ import annotations

import logging
from functools import cached_property

from langchain_community.vectorstores import Neo4jVector

from legal_assistant import config
from legal_assistant.rag.intent_classifier import QueryClassifier
from legal_assistant.rag.retrievers.hyde import HyDEGenerator
from legal_assistant.resources import make_chat_llm, make_embeddings, make_langchain_graph

logger = logging.getLogger(__name__)


class RagContext:
    """Holds the shared graph, vector stores, classifier and LLM clients."""

    def __init__(self) -> None:
        logger.info("[RagContext] Building shared resources…")
        self.graph = make_langchain_graph()

        # One embedding instance shared by every vector store. Must match the model used at
        # graph-build time (see pipelines/graph_build.py); both go through make_embeddings.
        self.embeddings = make_embeddings()

        self.article_vector_store = self._vector_store("Article", ["text", "id", "title"])

        self.classifier = QueryClassifier(graph=self.graph, llm=make_chat_llm())
        self.synthesis_llm = make_chat_llm()
        self.filter_llm = make_chat_llm()
        # Cheap LLM for the optional pre-synthesis context-curation stage.
        self.curator_llm = make_chat_llm()
        logger.info("[RagContext] Ready.")

    def _vector_store(self, label: str, text_properties: list[str]) -> Neo4jVector:
        """Open the existing Neo4j vector index for ``label``.

        The index itself is created at graph-build time; this only binds to it. Index name
        and node label are always the same string, so one argument covers both.
        """
        logger.info("[RagContext] Opening %s vector store…", label)
        return Neo4jVector.from_existing_graph(
            embedding=self.embeddings,
            url=config.NEO4J_URI,
            username=config.NEO4J_USERNAME,
            password=config.NEO4J_PASSWORD,
            index_name=label,
            node_label=label,
            text_node_properties=text_properties,
            embedding_node_property="textEmbedding",
        )

    @cached_property
    def paragraph_vector_store(self) -> Neo4jVector:
        """Paragraph-level store required by the topics retriever.

        Built lazily on first use so methods that don't need it (e.g. the hybrid
        retriever and the batch pipeline) don't pay its construction cost.
        """
        return self._vector_store("Paragraph", ["text", "id"])

    @cached_property
    def case_law_vector_store(self) -> Neo4jVector:
        """Judgment-paragraph store, used only on the INTERPRETIVE branch.

        Built lazily: a corpus with no case law ingested (see
        :mod:`legal_assistant.pipelines.case_law_ingest`) has no CaseLawParagraph vector
        index, and DEFINITIONAL queries never touch this store.
        """
        return self._vector_store("CaseLawParagraph", ["text", "id"])

    def make_hyde_generator(self, iterations: int) -> HyDEGenerator:
        """Build a HyDE generator; temperature > 0 only when sampling >1 doc so
        the hypothetical passages diverge."""
        hyde_llm = make_chat_llm(
            model="gpt-4o-mini", temperature=0.7 if iterations > 1 else 0.0
        )
        return HyDEGenerator(llm=hyde_llm, iterations=iterations)
