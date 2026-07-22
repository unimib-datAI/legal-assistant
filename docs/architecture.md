# Architecture

How the system is put together, and which pieces you extend rather than copy.

Extending the system — a new retrieval strategy, act, or pipeline — is covered separately
in [extending.md](extending.md). For the Neo4j schema see
[knowledge-graph.md](knowledge-graph.md); for command flags see the [README](../README.md).

## Tech stack

| Layer | Technology | Role |
|---|---|---|
| Language | Python 3.11+ | Entire codebase |
| Graph DB | Neo4j (Docker) | Nodes, relationships, and COSINE vector indexes |
| LLM orchestration | LangChain + `langchain-neo4j` | Retriever interface, prompt templates, `Neo4jVector` |
| LLM / Embeddings | OpenAI (`langchain-openai`) | Classification, synthesis, attribution, embeddings |
| Sparse retrieval | `rank-bm25` | BM25 over the act-scoped article and recital corpus |
| Reranking | `sentence-transformers` CrossEncoder | Cross-encoder rerank of fused candidates |
| Topic similarity | `sentence-transformers` (`all-MiniLM-L6-v2`) | ASKE topic matching in the `topics` retriever |
| NLP | NLTK | Tokenization, lemmatization, sentence splitting |
| HTML parsing | BeautifulSoup4 | EUR-Lex act and judgment markup |
| Browser | Playwright (Chromium) | Headless fetch that clears the AWS WAF challenge |
| Evaluation | RAGAS | LLM-judged answer metrics |
| UI | Streamlit | Pipeline controls and the evaluation viewer |
| Config | `python-dotenv` | `.env` → `config.py` |

## Layout

The project is an installable package (`pip install -e .`) under `src/`:

```
src/legal_assistant/       the package — everything importable
  config.py                env-driven settings (models, thresholds, paths)
  resources.py             the only place Neo4j / OpenAI clients are constructed
  logging_setup.py         the only place root logging is configured
  graph/                   Neo4j client, loader, Cypher queries, ASKE seeds
  scraper/                 EUR-Lex fetching and HTML parsing
  case_law/                CJEU judgment parsing and KG building
  text/  topic/            preprocessing and the ASKE algorithm
  rag/                     retrieval and answer synthesis
  pipelines/               the runnable jobs: graph_build, aske_run, case_law_ingest, summaries
  cli/                     `legal-assistant` argument parsing, nothing else
frontend/                  Streamlit app — thin UI shells over pipelines/
evals/                     eval harnesses, datasets, and result CSVs
tests/                     pytest unit tests (mirrors the package layout)
```

Inside `rag/`:

| Module | Role |
|---|---|
| `methods/base.py` | The `RagMethod` contract: `param_specs()` + `build_retriever()` |
| `methods/registry.py` | The list of available strategies |
| `retrievers/` | The retrieval implementations (`hybrid`, `topics`, `hyde`) |
| `context.py` | `RagContext` — expensive shared resources, built once |
| `documents.py` | Passage ids and source headers, shared by every retriever |
| `intent_classifier.py` | Query intent, type, and target-act selection |
| `attribution.py` | Post-hoc per-sentence `[Sn]` source attribution |
| `pipeline.py` | `RAGPipeline` — retrieve, curate, synthesise, filter |
| `prompts/` | Versioned prompts grouped by stage |

## Layers

```
cli/  ·  frontend/  ·  evals/          entry points — parse input, render output
        │
        ▼
pipelines/                             the runnable jobs (graph_build, aske_run,
        │                              case_law_ingest, summaries)
        ▼
rag/  graph/  scraper/  case_law/      domain logic
  text/  topic/
        │
        ▼
config.py  ·  resources.py  ·  logging_setup.py        foundations
```

The dependency rule is one-directional: **foundations know nothing about domain logic,
domain logic knows nothing about pipelines, and nothing imports an entry point.** Two
concrete invariants enforce it:

- **`resources.py` is the only place external clients are constructed.** No module calls
  `Neo4jGraph(...)`, `ChatOpenAI(...)` or `OpenAIEmbeddings(...)` directly; they call
  `make_graph_client()`, `make_langchain_graph()`, `make_chat_llm()`, `make_embeddings()`.
  That is what makes swapping an endpoint, a model, or a mock in tests a one-file change.
- **`logging_setup.py` is the only place root logging is configured.** Library modules do
  `logger = logging.getLogger(__name__)` and nothing else; entry points call
  `configure_logging()`. Importing the package never reconfigures a caller's logging.

`frontend/` sits outside the package because Streamlit's multipage API needs real file
paths, and because it is an application, not a library. Its pages hold widgets and
rendering only — every page delegates to `pipelines/`.

## Data flow

Three phases, each a CLI subcommand over a function in `pipelines/`.

### Phase 1 — Graph build (`pipelines/graph_build.py`)

`BrowserFetcher` renders each EUR-Lex page headless → `scraper/eurlex_exporter.py` parses
the hierarchy → `scraper/metadata_parser.py` extracts the "Interpreted by" case-law
references → `graph/loader.py` writes nodes and edges with parameterized Cypher →
Paragraph, Recital and Article nodes are embedded and given a vector index.

CJEU judgments are a separate job (`pipelines/case_law_ingest.py`) because they are fetched
per judgment and fail individually: `case_law/html_parser.py` reads the hierarchy straight
from the published XHTML, `case_law/tree.py` flattens it, `case_law/kg_builder.py` writes
it. A judgment that cannot be fetched is skipped and reported, never fatal.

### Phase 2 — Topic extraction (`pipelines/aske_run.py`)

Paragraphs are fetched from Neo4j and `text/preprocessor.py` tokenizes them into sentences,
lemmatizes each word, and produces sentence-level chunks (the first sentence is skipped —
it usually only holds the paragraph number). `topic/aske.py` then runs the ASKE cycle
against the seed concepts in `graph/seed.py` for `--generations` iterations, each with four
phases:

- **Chunk classification** — cosine similarity between chunk and concept embeddings; chunks
  above `α` are assigned to that concept.
- **Deactivate unused** — concepts with zero classifications are marked inactive and
  excluded from further enrichment.
- **Terminology enrichment** — candidate terms are extracted from the classified chunks with
  TF-IDF over bigrams, WordNet definitions are embedded, and terms are scored with a
  discriminative penalty (`sim_to_concept − 0.5 × max_sim_to_others`) that down-ranks
  generic terms. The top `γ` terms above `β` join the concept.
- **Concept derivation** — terms within a concept are clustered by affinity propagation
  (`topic/concept.py`); each distinct cluster spawns a new concept labelled by its centroid.

Finally the top-3 topics per paragraph are aggregated by maximum chunk-level score and
written back as Concept nodes linked to the paragraphs.

Only the `topics` retriever consumes this; `hybrid` ignores it entirely.

### Phase 3 — Query (`rag/pipeline.py`)

```
question
   │
   ▼  rag/intent_classifier.py
intent (DEFINITIONAL | INTERPRETIVE) · query_type · target acts
   │
   ▼  the selected RagMethod builds its retriever from RagContext
passages
   │
   ▼  rag/documents.py  — bracketed source header per passage
   ▼  [optional] context curation — a cheap LLM filters, never rewrites, fails open
   ▼  answer_synthesis prompt (active version) → answer
   ▼  [optional] answer filter — trim to the responsive sentences
answer + source ids
```

Attribution (`rag/attribution.py`) is a separate post-hoc pass: it splits the finished
answer locally with NLTK and asks an LLM only for `[Sn]` marker assignments, so the answer
text is never altered. `eval backfill` runs it over an existing results CSV.

---

## Testing

Tests mirror the package: `src/legal_assistant/rag/acts.py` → `tests/rag/test_acts.py`.
Mock Neo4j and OpenAI at the boundary with `unittest.mock`; `resources.py` is the seam that
makes that a single patch point.

The pure logic worth pinning first, none of which needs a database: `rag/acts.py`,
`rag/citations.py`, `rag/documents.py`, `rag/prompts/registry.py`, and the act-selection
half of `rag/intent_classifier.py`.
