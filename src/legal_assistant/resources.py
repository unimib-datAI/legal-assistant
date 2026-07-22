"""Factories for the external clients every entry point needs.

Neo4j connections, embedding clients, and chat models were previously constructed
inline in each script, page, and eval — each copy repeating the model id, the
``temperature``, and the ``base_url or None`` normalization. They are built here
instead, so a change to how the project talks to Neo4j or OpenAI is a one-file change.

Two distinct graph clients exist and both are needed:

* :func:`make_graph_client` returns the project's own :class:`legal_assistant.graph.client.Neo4jGraph`
  — a raw driver wrapper with the ingestion helpers (embedding generation, index creation).
* :func:`make_langchain_graph` returns LangChain's ``Neo4jGraph``, which is what the
  retrievers and ``Neo4jVector`` expect.
"""
from __future__ import annotations

from langchain_neo4j import Neo4jGraph as LangChainNeo4jGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from legal_assistant import config
from legal_assistant.graph.client import Neo4jGraph


def make_graph_client() -> Neo4jGraph:
    """The project's own Neo4j wrapper, used by the loaders and ingestion pipelines."""
    return Neo4jGraph(config.NEO4J_URI, config.NEO4J_USERNAME, config.NEO4J_PASSWORD)


def make_langchain_graph() -> LangChainNeo4jGraph:
    """LangChain's Neo4j graph, used by the retrievers and the query classifier."""
    return LangChainNeo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
    )


def make_embeddings(model: str | None = None) -> OpenAIEmbeddings:
    """The embedding client.

    Must be the same model at graph-build time and at query time, hence the single
    default from ``config.EMBEDDING_MODEL``.
    """
    return OpenAIEmbeddings(
        model=model or config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
        # An empty string is passed to the OpenAI client verbatim (it falls back only on
        # None, not on falsy) and breaks the URL, so normalize "" -> None to hit the
        # default endpoint.
        base_url=config.OPENAI_BASE_URL or None,
    )


def make_chat_llm(model: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
    """A chat model. ``temperature`` is always explicit — never left to the default."""
    return ChatOpenAI(
        model=model or config.RAG_LLM_MODEL,
        temperature=temperature,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL or None,
    )
