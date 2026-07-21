"""Graph-topic + vector RAG method (ASKE topic filtering)."""
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.retrievers import BaseRetriever

from service.rag.methods.base import ParamSpec, RagMethod
from service.rag.methods.context import RagContext
from service.rag.rag_naive_with_topics import GraphEnrichedRetriever


class TopicsRagMethod(RagMethod):
    id = "topics"
    name = "Graph topics + vector"
    description = (
        "Re-ranks paragraph candidates by semantic similarity to ASKE-extracted "
        "graph topics, combined with vector similarity search."
    )

    def param_specs(self) -> List[ParamSpec]:
        return [
            ParamSpec("use_topic_filter", "Use topic filter", "bool", True,
                      help="Enrich retrieval with paragraphs matched to graph topics."),
            ParamSpec("k", "Top-k", "int", 5, min=1, max=15, step=1),
            ParamSpec("top_k_topic", "Top-k topics", "int", 5, min=1, max=15, step=1),
            ParamSpec("topic_similarity_threshold", "Topic similarity threshold", "float", 0.35,
                      min=0.0, max=1.0, step=0.05,
                      help="Minimum cosine similarity for a topic to match the query."),
        ]

    def build_retriever(self, ctx: RagContext, config: Dict[str, Any]) -> BaseRetriever:
        return GraphEnrichedRetriever(
            vector_store=ctx.paragraph_vector_store,
            graph=ctx.graph,
            classifier=ctx.classifier,
            **config,
        )
