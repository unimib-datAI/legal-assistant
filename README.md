# Legal Assistant

A Graph-RAG system over EU digital regulation. It models a knowledge graph from four acts —
**GDPR**, **Data Governance Act**, **Data Act** and **AI Act** — plus the CJEU judgments
that interpret them, and answers legal questions grounded in those documents, traceable to
the article or paragraph that supports each claim.

All data comes from EUR-Lex: the acts are parsed from the published English HTML into a
structured hierarchy, and each act's document-information page is parsed to link the case
law that interprets it. Only completed case law is considered — ongoing cases are ignored.

- **[docs/architecture.md](docs/architecture.md)** — layout, layers, and how a query becomes
  an answer
- **[docs/extending.md](docs/extending.md)** — adding a retrieval strategy, an act, a
  pipeline
- **[docs/knowledge-graph.md](docs/knowledge-graph.md)** — the Neo4j schema: nodes, fields,
  and how they relate
- **[docs/aske.md](docs/aske.md)** — the topic-extraction algorithm

## Getting Started

```bash
pip install -e .            # installs the `legal_assistant` package and the CLI
docker compose up -d        # Neo4j
cp .env.example .env        # NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD / OPEN_API_KEY
```

Then build the graph and ask a question:

```bash
legal-assistant graph build
legal-assistant graph aske
legal-assistant ingest case-law --acts 32016R0679
legal-assistant rag query "What entities fall under the personal scope of Chapter II?"
```

`streamlit run frontend/app.py` opens the same pipelines behind a UI, plus the
RAG-vs-Ground-Truth evaluation viewer.

---

## Command Reference

Everything runs through the `legal-assistant` CLI. `--help` works at every level
(`legal-assistant rag query --help`). Global flag: `-v` / `--verbose` raises logging to
DEBUG.

### `graph build` — Phase 1: load the acts

Scrapes EUR-Lex, writes the graph, embeds and indexes Paragraph, Recital and Article nodes.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--celex` | list | the four acts | CELEX ids to load |
| `--no-clear` | flag | off | Keep the existing database instead of wiping it first |

```bash
legal-assistant graph build                                  # all four acts, fresh database
legal-assistant graph build --celex 32016R0679               # GDPR only
legal-assistant graph build --celex 32024R1689 --no-clear    # add the AI Act to what exists
```

Default acts: `32016R0679` (GDPR), `32024R1689` (AI Act), `32023R2854` (Data Act),
`32022R0868` (Data Governance Act).

### `graph aske` — Phase 2: topic extraction

Runs the ASKE cycle over the paragraphs in the graph and writes Concept nodes back.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--generations` | int | `15` | ASKE iterations |
| `--alpha` | float | `0.4` | Chunk-classification similarity threshold |
| `--beta` | float | `0.4` | Terminology-enrichment acceptance threshold |
| `--gamma` | int | `7` | Max new terms added per concept per generation |
| `--out` | path | — | Write the active-concept report as JSON |

```bash
legal-assistant graph aske
legal-assistant graph aske --generations 20 --alpha 0.3 --out results/aske.json
```

> Required only by the `topics` retrieval method. The `hybrid` method does not read
> Concept nodes, so this phase can be skipped entirely if you only use `hybrid`.

### `ingest case-law` — CJEU judgments

Reads the judgments to fetch out of the graph itself (the `(:CaseLaw)` stubs `graph build`
created), parses each from EUR-Lex XHTML, and embeds the paragraphs.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--acts` | list | `32016R0679` | Acts whose case law to ingest |
| `--celex` | list | — | Ingest these judgments explicitly instead of reading them from the graph |
| `--limit` | int | — | Ingest at most N judgments (smoke run) |
| `--reset` | flag | off | Delete existing sections/paragraphs first (`(:CaseLaw)` stubs and INTERPRETS edges are kept) |
| `--summaries` | flag | off | Generate LLM section summaries — one call per section |
| `--skip-embeddings` | flag | off | Do not embed or (re)create the vector index |

```bash
legal-assistant ingest case-law                                  # all GDPR case law
legal-assistant ingest case-law --acts 32016R0679 32024R1689
legal-assistant ingest case-law --celex 62019CJ0645 --summaries
legal-assistant ingest case-law --limit 3 --skip-embeddings      # quick smoke run
legal-assistant ingest case-law --reset                          # rebuild content from scratch
```

> Requires `graph build` to have run first. Judgments older than ~2012 have no XHTML
> manifestation in Cellar and are skipped, not fatal — the run lists them at the end.

### `ingest obligations` — deontic obligations

Detects candidate passages, filters and analyses them with `EXTRACTION_LLM_MODEL`
(`gpt-5-mini`), anchors addressees to the actor vocabulary, and writes the obligation
subgraph. Passages are processed concurrently (`EXTRACTION_MAX_WORKERS`, default 8) with
backoff on rate limits.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--acts` | list | `32016R0679` | CELEX ids to extract obligations from |
| `--limit` | int | — | Process at most N candidate passages (smoke run) |
| `--reset` | flag | off | Delete existing obligations for these acts first |

```bash
legal-assistant ingest obligations --acts 32016R0679 --reset     # full GDPR, clean rebuild
legal-assistant ingest obligations --acts 32024R1689 --limit 20  # AI Act smoke run
```

> Requires `graph build`. Actors promoted from addressee strings are written back into
> `obligations/actors.yaml` for review.

### `checklist` — every obligation a role bears

Deterministic: fetches the complete obligation set for a role (hierarchy included) and has the
LLM render it as a compliance checklist. Nothing is retrieved or truncated.

```bash
legal-assistant checklist --act 32016R0679 --actor controller
```

### `summarize` — optional LLM summaries on graph nodes

Idempotent: only nodes whose `summary` is still NULL are fetched, so a re-run on a fully
summarised graph exits immediately.

| Argument | Values | Default | Meaning |
|---|---|---|---|
| `kind` | `articles` \| `chapters` | required | Which nodes to summarise |
| `--concurrency` | int | `5` | Parallel LLM calls |

```bash
legal-assistant summarize articles
legal-assistant summarize chapters --concurrency 10
```

### `rag query` / `rag batch` — Phase 3: ask questions

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--method` | str | `hybrid` | Retrieval strategy id (`hybrid`, `topics`) |
| `--override` | `KEY=VALUE …` | — | Override any of the method's own params |
| `--answer-filter` | flag | off | Post-filter the draft answer to the responsive sentences |
| `--context-curation` | flag | off | Curate retrieved passages with a cheap LLM before synthesis |
| `--prompt-version` | str | active (`v8`) | Pin a registered `answer_synthesis` version |

`rag batch` adds `--questions <file.json>` and `--out <report.json>`, both required.

```bash
legal-assistant rag query "What is personal data?"
legal-assistant rag query "..." --method topics
legal-assistant rag query "..." --override use_case_law=false top_k_final=8
legal-assistant rag query "..." --prompt-version v9 --context-curation
legal-assistant rag batch --questions corpus/golden_dataset.json --out results/report.json
```

`--override` values are parsed as JSON, so `false`, `8` and `0.3` arrive as the right
Python types. An unknown key fails fast with the list of valid ones.

### `rag methods` — what is registered

Prints every strategy with its description and every tunable parameter (name, type,
default). This is the authoritative list of what `--override` accepts.

```bash
legal-assistant rag methods
```

`hybrid` exposes 18 params (`use_hyde`, `hyde_iterations`, `top_k_dense`, `top_k_sparse`,
`top_k_final`, `top_k_recitals`, `recital_score_threshold`, `use_recitals`, `use_case_law`,
`top_k_case_law`, `case_law_score_threshold`, `case_law_neighbours`, `guarantee_operative`,
`top_k_bridge`, `use_query_decomposition`, `max_sub_questions`, `rrf_k`, `use_reranker`);
`topics` exposes 4 (`use_topic_filter`, `k`, `top_k_topic`, `topic_similarity_threshold`).

---

## Evaluation

`legal-assistant eval <harness> [flags]` runs a harness from `evals/`, passing the flags
straight through. Each harness also runs directly (`python evals/<script>.py`). Datasets
live in `evals/evals/datasets/`, results land in `evals/evals/evaluations/`, and full run
logs in `evals/evals/logs/`.

Available datasets: `golden_dataset`, `golden_dataset_light`, `case_law_golden_dataset`,
`subset_retrieval_scarso`, `subset_retrieval_scarso_dga`, `question_recital_required`,
`zero_tp_queries`.

### `eval retrieval` — deterministic retrieval quality

Scores only the retrieval step against the provisions each ground-truth answer cites
inline. No answer synthesis, no LLM judge — cheap and, HyDE sampling aside, deterministic.
Metrics: `top1_anchor`, `recall_at_k`, `mrr`, aggregated overall and per act.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--method` | str | `hybrid` | RAG method id |
| `--dataset` | str | `subset_retrieval_scarso` | Dataset name, without `.csv` |
| `--repeats` | int | `3` | Runs per query, averaged to wash out HyDE variance |
| `--no-case-law` | flag | off | Disable the INTERPRETIVE case law branch — the control arm |

```bash
legal-assistant eval retrieval
legal-assistant eval retrieval --dataset golden_dataset --repeats 1
legal-assistant eval retrieval --no-case-law                          # A/B the case law branch
RERANK_MODEL=BAAI/bge-reranker-v2-m3 legal-assistant eval retrieval   # A/B the reranker
```

> The right harness for A/B-ing retrieval changes: no generation and no judge, so a
> difference in the numbers is a real difference in retrieval.

### `eval ragas` — end-to-end answer quality

Runs the full pipeline and scores the generated answer with six RAGAS metrics:
`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`,
`factual_precision`, `factual_recall`.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--method` | str | `hybrid` | RAG method id |
| `--dataset` | str | `case_law_golden_dataset` | Dataset name, without `.csv` |
| `--curate` / `--no-curate` | flag | **on** | Pre-synthesis context curation |
| `--decompose` | flag | off | Query decomposition into sub-questions |
| `--synthesis-prompt` | str | active (`v8`) | Pin a registered `answer_synthesis` version |

```bash
legal-assistant eval ragas
legal-assistant eval ragas --dataset golden_dataset --method topics
legal-assistant eval ragas --synthesis-prompt v9 --no-curate    # isolate the prompt change
```

> `--curate` defaults to **on** here, unlike `RAGPipeline` itself. When A/B-ing a prompt
> version, pass the same curation setting on both arms or you are moving two variables at
> once. The version used is written into the output filename and into the
> `synthesis_prompt_version` column of every row, so runs stay distinguishable.

### `eval acts` — act-classification accuracy

Runs only `QueryClassifier.classify` — no retrieval, no synthesis, no reranker — so it is
cheap. Reports `detected` (true act in the prediction), `exact`, and `abstained`.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--dataset` | str | `golden_dataset` | Dataset name; needs an `act` column |
| `--threshold` | float | `ACT_SCORE_THRESHOLD` (`0.4`) | Per-act relevance threshold |
| `--limit` | int | — | Classify only the first N rows |

```bash
legal-assistant eval acts
legal-assistant eval acts --threshold 0.5
legal-assistant eval acts --limit 5                     # smoke run
```

### `eval roles` — obligation role-recall

Generated from the graph, no annotation: for every actor that bears obligations, asks
"What obligations does a `<role>` have?" and checks the addressee classifier picks that role
and the graph returns a non-empty set. Needs only the classifier LLM and Neo4j.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--act` | str | `32016R0679` | CELEX id of the act to evaluate |
| `--limit` | int | — | Evaluate at most N roles |

```bash
legal-assistant eval roles --act 32016R0679 --limit 10
```

### `eval backfill` — add attribution to an existing results CSV

Adds per-sentence source attribution to a results CSV *post hoc* — one LLM call per row,
no re-retrieval, no re-synthesis. Appends `segments` and `cited_sources` (both JSON); the
frontend's evaluation page renders them as `[Sn]` badges.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--csv` | path | required | Results CSV to enrich |
| `--out` | path | overwrite `--csv` | Output path |
| `--model` | str | `RAG_LLM_MODEL` | Chat model for the attribution pass |

```bash
legal-assistant eval backfill --csv evals/evals/evaluations/best_bench.csv \
                              --out evals/evals/evaluations/best_bench_attributed.csv
```

> Requires the CSV to have `answer`, `sources` and `contexts` columns.

> `evals/evals.py` is the older custom-metric harness. It takes no flags — dataset and
> options are hardcoded — and is kept for reference only; use `eval ragas` instead.

---

## Configuration

All settings are read from `.env` (see `src/legal_assistant/config.py`).

| Variable | Default | Meaning |
|---|---|---|
| `NEO4J_URI` | — | Neo4j bolt URI |
| `NEO4J_USERNAME` / `NEO4J_PASSWORD` | — | Neo4j credentials |
| `OPEN_API_KEY` | — | OpenAI API key |
| `OPENAI_BASE_URL` | `""` | Alternative endpoint; empty means the default |
| `RAG_LLM_MODEL` | `gpt-4o-mini` | Chat model: classifier, synthesis, filter, curation |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Must match at build and query time |
| `EMBEDDING_DIM` | `1536` | Must match the embedding model's output size |
| `ACT_SCORE_THRESHOLD` | `0.4` | Minimum per-act relevance for an act to become a retrieval target |
| `RERANK_MODEL` | `zeroentropy/zerank-2` | Cross-encoder reranker |
| `RERANK_TRUST_REMOTE_CODE` | `false` | Needed by rerankers shipping custom modeling code |
| `RERANK_DTYPE` | `auto` | `auto` respects the checkpoint's native dtype (bf16 ≈ half the VRAM) |

Changing `EMBEDDING_MODEL` or `EMBEDDING_DIM` invalidates the stored vectors — re-run
`graph build` after either.
