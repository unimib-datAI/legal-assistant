"""Hybrid (dense + BM25 + RRF + HyDE) RAG method."""
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.retrievers import BaseRetriever

from service.rag.methods.base import ParamSpec, RagMethod
from service.rag.methods.context import RagContext
from service.rag.rag_alternative import HybridRetriever

# Fields consumed when building the HyDE generator rather than passed straight
# through to the retriever constructor.
_HYDE_ITERATIONS = "hyde_iterations"


class HybridRagMethod(RagMethod):
    id = "hybrid"
    name = "Hybrid (Dense + BM25 + RRF + HyDE)"
    description = (
        "Dense vector + sparse BM25 article search fused with Reciprocal Rank "
        "Fusion, optional HyDE query expansion, and cross-encoder reranking."
    )

    def param_specs(self) -> List[ParamSpec]:
        return [
            ParamSpec("use_hyde", "Use HyDE", "bool", True,
                      help="Expand the query into hypothetical legal passages before dense search."),
            ParamSpec(_HYDE_ITERATIONS, "HyDE iterations", "int", 3, min=1, max=5, step=1,
                      help="Number of hypothetical passages to sample and average."),
            ParamSpec("top_k_dense", "Top-k dense", "int", 10, min=1, max=30, step=1),
            ParamSpec("top_k_sparse", "Top-k sparse (BM25)", "int", 10, min=1, max=30, step=1),
            ParamSpec("top_k_final", "Top-k final articles", "int", 5, min=1, max=15, step=1),
            ParamSpec("top_k_recitals", "Top-k recitals", "int", 2, min=0, max=10, step=1),
            ParamSpec("recital_score_threshold", "Recital score threshold", "float", 0.3,
                      min=0.0, max=1.0, step=0.05,
                      help="Minimum cross-encoder score for a recital to be kept."),
            ParamSpec("use_recitals", "Search recitals", "bool", True,
                      help="Retrieve and rerank recitals alongside articles; when off, only articles are returned."),
            ParamSpec("use_case_law", "Search case law (INTERPRETIVE)", "bool", True,
                      help="On INTERPRETIVE queries, search CJEU judgment paragraphs and boost the "
                           "articles those judgments interpret. No effect on DEFINITIONAL queries."),
            ParamSpec("top_k_case_law", "Top-k case law paragraphs", "int", 5, min=0, max=10, step=1),
            ParamSpec("case_law_score_threshold", "Case law score threshold", "float", 0.3,
                      min=0.0, max=1.0, step=0.05,
                      help="Minimum cross-encoder score for a judgment paragraph to be kept."),
            ParamSpec("case_law_neighbours", "Case law neighbour window", "int", 2, min=0, max=5, step=1,
                      help="Paragraphs read after each judgment hit (and one before) to recover the "
                           "holding, which follows the reasoning and is too terse to match on its "
                           "own. 0 disables the expansion."),
            ParamSpec("guarantee_operative", "Guarantee the operative part", "bool", True,
                      help="Give the operative part of the top-ranked judgment a slot of its own "
                           "when it clears the threshold and no operative passage was retrieved."),
            ParamSpec("top_k_bridge", "Top-k bridged articles", "int", 3, min=0, max=10, step=1,
                      help="Articles cited in the text of the retrieved judgment passages, fused "
                           "into the article ranking. 0 disables the graph boost."),
            ParamSpec("use_query_decomposition", "Decompose into sub-questions", "bool", False,
                      help="Split compound questions into sub-questions and retrieve (HyDE+BM25) per sub-question."),
            ParamSpec("max_sub_questions", "Max sub-questions", "int", 3, min=1, max=6, step=1,
                      help="Cap on the number of decomposed sub-questions used for retrieval."),
            ParamSpec("rrf_k", "RRF k", "int", 60, min=1, max=120, step=1,
                      help="Reciprocal Rank Fusion constant."),
            ParamSpec("use_reranker", "Use cross-encoder reranker", "bool", True),
        ]

    def build_retriever(self, ctx: RagContext, config: Dict[str, Any]) -> BaseRetriever:
        cfg = dict(config)
        use_hyde = cfg.get("use_hyde", True)
        iterations = cfg.pop(_HYDE_ITERATIONS, 3)
        hyde_generator = ctx.make_hyde_generator(iterations) if use_hyde else None
        # The case law store is a cached_property: touching it here builds it, so only reach
        # for it when the branch is actually enabled.
        case_law_store = ctx.case_law_vector_store if cfg.get("use_case_law", True) else None

        return HybridRetriever(
            graph=ctx.graph,
            article_vector_store=ctx.article_vector_store,
            case_law_vector_store=case_law_store,
            classifier=ctx.classifier,
            hyde_generator=hyde_generator,
            **cfg,
        )
