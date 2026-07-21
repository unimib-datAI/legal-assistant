"""Batch-ingest CJEU case law into the knowledge graph.

The judgments to ingest are read straight out of the graph: the act loader already creates a
(:CaseLaw) stub per judgment listed as "Interpreted by" in the EUR-Lex metadata, together with
the (:CaseLaw)-[:INTERPRETS]->(:Article|:Paragraph|:Chapter) edge. Those stubs *are* the set of
judgments interpreting our acts, so this script needs no separate case list.

Run the act loader (frontend/kg/graph_init.py) first, then::

    python case_law_init.py                 # all judgments interpreting the GDPR
    python case_law_init.py --reset         # rebuild the case law content from scratch
    python case_law_init.py --acts 32016R0679 32024R1689
    python case_law_init.py --celex 62019CJ0645 --summaries

Summaries are off by default: they cost one LLM call per section and paragraph-level retrieval
does not use them.
"""
import argparse
import logging
import sys

from langchain_openai import OpenAIEmbeddings

import config
from service.case_law.html_parser import CaseLawHTMLError, parse_celex
from service.case_law.kg_builder import build_from_tree, celex_to_case_number
from service.case_law.tree import flatten
from service.graph.graph import Neo4jGraph
from service.graph.query import CaseLawQueries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("case_law_init")

# Chatty at INFO and it drowns the per-judgment progress: every node upsert and every edge.
logging.getLogger("service.graph.graph").setLevel(logging.WARNING)

_NODE_LABEL = "CaseLawParagraph"


def _resolve_celex_list(graph: Neo4jGraph, args: argparse.Namespace) -> list[str]:
    if args.celex:
        return [c.strip().upper() for c in args.celex]

    rows = graph.query(CaseLawQueries.GET_CASE_LAW_BY_ACTS, {"acts": args.acts})
    celex_list = [row["celex"] for row in rows]
    if args.limit:
        celex_list = celex_list[: args.limit]
    return celex_list


def _summarize(celex: str, flat: list[dict]) -> list[dict]:
    from service.case_law.llm_orchestrator import summarize_document, summarize_section

    logger.info("Summarising %s (%d sections)…", celex, len(flat))
    summaries = [s for s in (summarize_section(section) for section in flat) if s]
    return summaries


def ingest(graph: Neo4jGraph, celex_list: list[str], with_summaries: bool) -> dict:
    """Parse and write each judgment. A judgment that cannot be fetched is skipped, not fatal."""
    totals = {"judgments": 0, "sections": 0, "paragraphs": 0, "operative": 0}
    failed: list[tuple[str, str]] = []

    for i, celex in enumerate(celex_list, start=1):
        label = f"{celex} ({celex_to_case_number(celex)})"
        try:
            roots = parse_celex(celex)
        except CaseLawHTMLError as exc:
            # Judgments before ~2012 have no XHTML manifestation in Cellar, only FMX.
            logger.warning("[%d/%d] SKIP %s — %s", i, len(celex_list), label, exc)
            failed.append((celex, str(exc)))
            continue
        except OSError as exc:
            logger.warning("[%d/%d] SKIP %s — fetch failed: %s", i, len(celex_list), label, exc)
            failed.append((celex, f"fetch failed: {exc}"))
            continue

        summaries = _summarize(celex, flatten(roots)) if with_summaries else None
        counts = build_from_tree(celex, roots, graph, summaries=summaries)

        totals["judgments"] += 1
        for key in ("sections", "paragraphs", "operative"):
            totals[key] += counts[key]
        logger.info(
            "[%d/%d] %s — %d sections, %d paragraphs (%d operative)",
            i, len(celex_list), label,
            counts["sections"], counts["paragraphs"], counts["operative"],
        )

    totals["failed"] = failed
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--acts", nargs="+", default=["32016R0679"],
                        help="CELEX ids of the acts whose case law to ingest (default: GDPR).")
    parser.add_argument("--celex", nargs="+",
                        help="Ingest these judgments explicitly instead of reading them from the graph.")
    parser.add_argument("--limit", type=int, help="Ingest at most N judgments (for a smoke run).")
    parser.add_argument("--reset", action="store_true",
                        help="Delete existing case law sections/paragraphs/topics before ingesting.")
    parser.add_argument("--summaries", action="store_true",
                        help="Generate LLM section summaries (one call per section; off by default).")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Do not embed or (re)create the vector index.")
    args = parser.parse_args()

    graph = Neo4jGraph(config.NEO4J_URI, config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
    graph.verify_connection()

    try:
        if args.reset:
            logger.info("Deleting existing case law content (CaseLaw nodes and their INTERPRETS edges are kept)…")
            graph.query(CaseLawQueries.DELETE_CASE_LAW_CONTENT)

        celex_list = _resolve_celex_list(graph, args)
        if not celex_list:
            logger.error(
                "No case law found for acts %s. Has the act loader run? The (:CaseLaw) stubs and "
                "their INTERPRETS edges are created by GraphLoader, not by this script.", args.acts
            )
            return 1

        logger.info("Ingesting %d judgment(s) for acts %s…", len(celex_list), args.acts)
        totals = ingest(graph, celex_list, with_summaries=args.summaries)

        if not args.skip_embeddings and totals["paragraphs"]:
            embeddings = OpenAIEmbeddings(
                model=config.EMBEDDING_MODEL,
                api_key=config.OPENAI_API_KEY,
                base_url=config.OPENAI_BASE_URL or None,
            )
            graph.generate_text_embeddings(
                embed_fn=embeddings.embed_documents,
                embedding_dim=config.EMBEDDING_DIM,
                node_name=_NODE_LABEL,
            )
            graph.create_vector_index(_NODE_LABEL, _NODE_LABEL, config.EMBEDDING_DIM)

        failed = totals["failed"]
        logger.info(
            "Done — %d/%d judgments, %d sections, %d paragraphs (%d operative), %d skipped.",
            totals["judgments"], len(celex_list), totals["sections"],
            totals["paragraphs"], totals["operative"], len(failed),
        )
        for celex, reason in failed:
            logger.info("  skipped %s: %s", celex, reason)
        return 0
    finally:
        graph.close()


if __name__ == "__main__":
    sys.exit(main())
