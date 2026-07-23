"""The ``legal-assistant`` command line.

Every subcommand is a thin wrapper over a function in :mod:`legal_assistant.pipelines`
or :mod:`legal_assistant.rag`: argument parsing lives here, logic lives there, so each
pipeline stays callable (and testable) without a shell.

    legal-assistant graph build            # phase 1: scrape EUR-Lex -> Neo4j -> embeddings
    legal-assistant graph aske             # phase 2: topic extraction
    legal-assistant ingest case-law        # CJEU judgments for the acts in the graph
    legal-assistant summarize articles     # optional: LLM summaries on graph nodes
    legal-assistant rag query "..."        # phase 3: ask a question
    legal-assistant eval retrieval         # run an eval harness from evals/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import runpy
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from legal_assistant import config
from legal_assistant.logging_setup import configure_logging, quiet

logger = logging.getLogger("legal_assistant.cli")

# Eval harnesses live in the repo's evals/ directory, not in the installed package:
# they are dev tooling that carries its own datasets and result CSVs.
_EVAL_SCRIPTS = {
    "retrieval": "retrieval_eval.py",
    "ragas": "evals_ragas.py",
    "acts": "act_classification_eval.py",
    "backfill": "backfill_attribution.py",
}


# ── graph ─────────────────────────────────────────────────────────────────────

def _cmd_graph_build(args: argparse.Namespace) -> int:
    from legal_assistant.pipelines.graph_build import DEFAULT_CELEX_IDS, build_graph

    celex_ids = args.celex or list(DEFAULT_CELEX_IDS)
    result = build_graph(celex_ids, clear_db=not args.no_clear, strict=not args.allow_invalid)
    logger.info(
        "Graph built, %d document(s), indexed: %s",
        len(result.celex_ids), ", ".join(result.indexed_labels),
    )
    return 0


def _cmd_graph_aske(args: argparse.Namespace) -> int:
    from legal_assistant.pipelines.aske_run import AskeParams, run_aske

    result = run_aske(AskeParams(
        n_generations=args.generations, alpha=args.alpha, beta=args.beta, gamma=args.gamma,
    ))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result.as_report(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Concept report written to %s", args.out)
    return 0


# ── ingest ────────────────────────────────────────────────────────────────────

def _cmd_ingest_case_law(args: argparse.Namespace) -> int:
    from legal_assistant.pipelines import case_law_ingest as ingest_mod
    from legal_assistant.resources import make_graph_client

    # Chatty at INFO (every node upsert and every edge), and it drowns per-judgment progress.
    quiet("legal_assistant.graph.client")

    graph = make_graph_client()
    graph.verify_connection()
    try:
        if args.reset:
            ingest_mod.delete_existing_content(graph)

        celex_list = ingest_mod.resolve_celex_list(graph, args.acts, args.celex, args.limit)
        if not celex_list:
            logger.error(
                "No case law found for acts %s. Has the act loader run? The (:CaseLaw) stubs "
                "and their INTERPRETS edges are created by `graph build`, not by this command.",
                args.acts,
            )
            return 1

        logger.info("Ingesting %d judgment(s) for acts %s…", len(celex_list), args.acts)
        totals = ingest_mod.ingest(
            graph, celex_list, with_summaries=args.summaries, strict=not args.allow_invalid
        )

        if not args.skip_embeddings and totals.paragraphs:
            ingest_mod.embed_and_index(graph)

        logger.info(
            "Done: %d/%d judgments, %d sections, %d paragraphs (%d operative), %d skipped.",
            totals.judgments, len(celex_list), totals.sections,
            totals.paragraphs, totals.operative, len(totals.failed),
        )
        for celex, reason in totals.failed:
            logger.info("  skipped %s: %s", celex, reason)

        # A skipped judgment is a failure, not a warning: exit non-zero so a scripted
        # ingest does not report success on a partial run.
        return 1 if totals.failed else 0
    finally:
        graph.close()


# ── summarize ─────────────────────────────────────────────────────────────────

def _cmd_summarize(args: argparse.Namespace) -> int:
    from legal_assistant.pipelines.summaries import TASKS, run_summaries
    from legal_assistant.resources import make_chat_llm, make_langchain_graph

    written = asyncio.run(run_summaries(
        make_langchain_graph(), make_chat_llm(), TASKS[args.kind], concurrency=args.concurrency,
    ))
    logger.info("Wrote %d %s summar(y/ies).", written, args.kind)
    return 0


# ── rag ───────────────────────────────────────────────────────────────────────

def _parse_overrides(pairs: Sequence[str] | None) -> Dict[str, Any]:
    """``--override use_case_law=false top_k_final=8`` -> a typed config dict.

    Values are read as JSON when possible (so ``false``/``8``/``0.3`` become the right
    Python types) and fall back to the raw string otherwise.
    """
    overrides: Dict[str, Any] = {}
    for pair in pairs or []:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise SystemExit(f"--override expects key=value, got {pair!r}")
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw
    return overrides


def _build_pipeline(args: argparse.Namespace):
    from legal_assistant.rag.pipeline import RAGPipeline

    return RAGPipeline(
        method_id=args.method,
        use_answer_filter=args.answer_filter,
        use_context_curation=args.context_curation,
        overrides=_parse_overrides(args.override) or None,
        synthesis_prompt_version=args.prompt_version,
    )


def _cmd_rag_query(args: argparse.Namespace) -> int:
    result = _build_pipeline(args).query(args.question)
    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"  - {source}")
    return 0


def _cmd_rag_batch(args: argparse.Namespace) -> int:
    _build_pipeline(args).run_batch(args.questions, args.out)
    return 0


def _cmd_rag_methods(args: argparse.Namespace) -> int:
    from legal_assistant.rag.methods.registry import list_methods

    for method in list_methods():
        print(f"{method.id:10s} {method.name}")
        print(f"{'':10s} {method.description}")
        for spec in method.param_specs():
            print(f"{'':12s} {spec.name} ({spec.kind}, default={spec.default})")
        print()
    return 0


# ── eval ──────────────────────────────────────────────────────────────────────

def _cmd_eval(args: argparse.Namespace) -> int:
    """Run one of the harnesses in ``evals/`` with its own arguments passed through."""
    script = config.PROJECT_ROOT / "evals" / _EVAL_SCRIPTS[args.harness]
    if not script.is_file():
        logger.error(
            "%s not found. The eval harnesses ship with the repository, not the installed "
            "package; run this from a checkout.", script,
        )
        return 1

    sys.argv = [str(script), *args.harness_args]
    runpy.run_path(str(script), run_name="__main__")
    return 0


# ── parser ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="legal-assistant", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true", help="Log at DEBUG level.")
    sub = parser.add_subparsers(dest="command", required=True)

    # graph
    graph = sub.add_parser("graph", help="Build the act knowledge graph and its topics.")
    graph_sub = graph.add_subparsers(dest="graph_command", required=True)

    build = graph_sub.add_parser("build", help="Scrape EUR-Lex, load Neo4j, embed and index.")
    build.add_argument("--celex", nargs="+", help="CELEX ids to load (default: the four acts).")
    build.add_argument("--no-clear", action="store_true",
                       help="Keep the existing database instead of wiping it first.")
    build.add_argument("--allow-invalid", action="store_true",
                       help="Write acts that fail graph validation, logging the violations "
                            "as warnings instead of aborting.")
    build.set_defaults(func=_cmd_graph_build)

    aske = graph_sub.add_parser("aske", help="Run the ASKE topic-extraction cycle.")
    aske.add_argument("--generations", type=int, default=15)
    aske.add_argument("--alpha", type=float, default=0.4, help="Classification threshold.")
    aske.add_argument("--beta", type=float, default=0.4, help="Enrichment threshold.")
    aske.add_argument("--gamma", type=int, default=7, help="Max new terms per concept.")
    aske.add_argument("--out", type=Path, help="Write the active-concept report as JSON.")
    aske.set_defaults(func=_cmd_graph_aske)

    # ingest
    ingest = sub.add_parser("ingest", help="Ingest external sources into the graph.")
    ingest_sub = ingest.add_subparsers(dest="ingest_command", required=True)

    case_law = ingest_sub.add_parser("case-law", help="Ingest CJEU judgments from EUR-Lex.")
    case_law.add_argument("--acts", nargs="+", default=["32016R0679"],
                          help="CELEX ids of the acts whose case law to ingest (default: GDPR).")
    case_law.add_argument("--celex", nargs="+",
                          help="Ingest these judgments explicitly instead of reading them from the graph.")
    case_law.add_argument("--limit", type=int, help="Ingest at most N judgments (for a smoke run).")
    case_law.add_argument("--reset", action="store_true",
                          help="Delete existing case law sections/paragraphs before ingesting.")
    case_law.add_argument("--summaries", action="store_true",
                          help="Generate LLM section summaries (one call per section; off by default).")
    case_law.add_argument("--skip-embeddings", action="store_true",
                          help="Do not embed or (re)create the vector index.")
    case_law.add_argument("--allow-invalid", action="store_true",
                          help="Write judgments that fail graph validation, logging the "
                               "violations as warnings instead of skipping them.")
    case_law.set_defaults(func=_cmd_ingest_case_law)

    # summarize
    summarize = sub.add_parser("summarize", help="Generate LLM summaries on graph nodes.")
    summarize.add_argument("kind", choices=("articles", "chapters"))
    summarize.add_argument("--concurrency", type=int, default=5)
    summarize.set_defaults(func=_cmd_summarize)

    # rag
    rag = sub.add_parser("rag", help="Query the RAG pipeline.")
    rag_sub = rag.add_subparsers(dest="rag_command", required=True)

    for name, help_text in (("query", "Answer one question."),
                            ("batch", "Answer a JSON file of questions.")):
        p = rag_sub.add_parser(name, help=help_text)
        p.add_argument("--method", default="hybrid",
                       help="RAG method id (see `legal-assistant rag methods`).")
        p.add_argument("--override", nargs="+", metavar="KEY=VALUE",
                       help="Override any of the method's params, e.g. use_case_law=false.")
        p.add_argument("--answer-filter", action="store_true", dest="answer_filter",
                       help="Post-filter the draft answer down to the responsive sentences.")
        p.add_argument("--context-curation", action="store_true", dest="context_curation",
                       help="Curate retrieved passages with a cheap LLM before synthesis.")
        p.add_argument("--prompt-version",
                       help="Pin a registered answer_synthesis version (e.g. v9) for an A/B.")

    query = rag_sub.choices["query"]
    query.add_argument("question")
    query.set_defaults(func=_cmd_rag_query)

    batch = rag_sub.choices["batch"]
    batch.add_argument("--questions", required=True, help="JSON file of questions.")
    batch.add_argument("--out", required=True, help="Where to write the JSON report.")
    batch.set_defaults(func=_cmd_rag_batch)

    methods = rag_sub.add_parser("methods", help="List the registered retrieval strategies.")
    methods.set_defaults(func=_cmd_rag_methods)

    # eval
    ev = sub.add_parser("eval", help="Run an evaluation harness from evals/.")
    ev.add_argument("harness", choices=tuple(_EVAL_SCRIPTS))
    ev.add_argument("harness_args", nargs=argparse.REMAINDER,
                    help="Arguments passed through to the harness (e.g. --dataset golden_dataset).")
    ev.set_defaults(func=_cmd_eval)

    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(logging.DEBUG if args.verbose else logging.INFO, show_logger_name=True)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
