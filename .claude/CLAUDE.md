# Legal Assistant Project

A Graph-RAG system over EU digital-regulation legislation. It models a knowledge graph from four acts — **AI Act**, **Data Governance Act**, **Data Act**, and **GDPR** — and exposes a retrieval-augmented generation (RAG) interface that answers legal questions grounded in those documents.

Queries are enriched by graph-aware topic filtering (ASKE) before being passed to an LLM, so answers are both semantically relevant and traceable to specific articles and paragraphs.

# Layout

The project is an installable package (`pip install -e .`) under `src/`:

| Path | Contents |
|---|---|
| `src/legal_assistant/` | The package — everything importable |
| `src/legal_assistant/config.py` | Env-driven settings: models, thresholds, paths |
| `src/legal_assistant/resources.py` | **The only** place Neo4j / OpenAI clients are constructed |
| `src/legal_assistant/logging_setup.py` | **The only** place root logging is configured |
| `src/legal_assistant/pipelines/` | The runnable jobs, callable without a shell |
| `src/legal_assistant/cli/` | `legal-assistant` argument parsing, nothing else |
| `frontend/` | Streamlit app — thin UI shells over `pipelines/` |
| `evals/` | Eval harnesses, datasets, result CSVs |
| `tests/` | pytest unit tests, mirroring the package layout |

Two rules that keep it that way:
- **Never construct `Neo4jGraph` / `ChatOpenAI` / `OpenAIEmbeddings` inline.** Use the
  factories in `resources.py` (`make_graph_client`, `make_langchain_graph`,
  `make_chat_llm`, `make_embeddings`).
- **Never call `logging.basicConfig` outside `logging_setup.py`.** Library modules only
  ever do `logger = logging.getLogger(__name__)`; entry points call `configure_logging()`.

`corpus/` holds the EUR-Lex source HTML the graph is built from — input data, not
documentation. Documentation lives in `docs/`.

# Documentation

Only this file is loaded automatically. The documents below are **not** — read the one that
matches the task before starting, rather than inferring the design from the code.

| Before… | Read |
|---|---|
| writing Cypher or touching `graph/` | `docs/knowledge-graph.md` |
| adding or changing a RAG strategy | `docs/extending.md` |
| adding an act, a pipeline, or a summarisation task | `docs/extending.md` |
| moving modules or changing package boundaries | `docs/architecture.md` |
| tracing how a query becomes an answer | `docs/architecture.md` |
| touching `topic/` or the ASKE cycle | `docs/aske.md` |
| changing a CLI flag or an eval harness | `README.md` |

## Keeping it current

One fact has one home. When a change lands, update the document that owns it — do not
restate it elsewhere, because the copy is what goes stale.

| When you change… | Update |
|---|---|
| `graph/loader.py` or `graph/queries.py` in a way that alters the schema | `docs/knowledge-graph.md` — **re-derive the fields from a live Neo4j** with `db.schema.nodeTypeProperties()`; never write them from memory or from the loader source |
| the `RagMethod` contract, the registry, or an extension point | `docs/extending.md` |
| a package folder, a layer boundary, or one of the two rules above | `docs/architecture.md` **and** the Layout table here |
| a CLI flag, its default, or an eval harness flag | the Command Reference in `README.md` |
| the ASKE algorithm or its parameters | `docs/aske.md` |

Documents describe **contracts and conventions**, not implementation detail the code already
states. A doc that repeats the code diverges from it; a doc that explains *why* does not.

# Python Coding Rules

## Style & Structure
- Follow PEP 8. Use `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep functions short and single-purpose. If a function does more than one thing, split it.
- Prefer flat over nested — early returns instead of deeply nested `if/else`.
- Use type hints on all function signatures. Avoid `Any` unless truly necessary.
- Use `dataclasses` or `pydantic.BaseModel` for structured data instead of raw dicts.
- Never write `—` (em dash) in comments, docstrings, log records, or UI text. Use a colon,
  a comma, or a full stop. Two places are exempt because there the character is data, not
  punctuation, and changing it changes behaviour: text that reaches the LLM (prompt
  templates in `rag/prompts/`, Pydantic field descriptions, the cited source headers built
  in `rag/documents.py` and `rag/attribution.py`), and regexes matching EUR-Lex markup
  (`_DASH` in `case_law/html_parser.py`). Comments quoting real source text keep it too.

## Error Handling
- Catch specific exceptions, never bare `except:` or `except Exception:` without re-raising.
- Fail fast at system boundaries (API calls, DB queries, file I/O). Let internal logic trust validated data.
- Use logging (`logging` module) instead of `print()` for diagnostics.

## Dependencies & Imports
- Group imports: stdlib, third-party, local — separated by blank lines.
- Never import with `*`. Use explicit imports.
- Pin dependency versions in `requirements.txt`.

## Data & Performance
- Use generators and lazy evaluation for large datasets — avoid loading everything into memory.
- Prefer list/dict comprehensions over manual loops for simple transforms.
- Use `pathlib.Path` over `os.path` for file system operations.

## Security
- Never hardcode secrets. Always load from environment variables or `.env`.
- Sanitize and validate all external input (user queries, API responses, file contents).
- Use parameterized queries for any database interaction — never string-format Cypher/SQL.

## Testing & Reliability
- Write tests for any non-trivial logic. Use `pytest` as the test runner.
- Use `unittest.mock` to isolate external dependencies (Neo4j, OpenAI API) in tests.
- Keep test files mirroring source structure: `src/legal_assistant/rag/foo.py` → `tests/rag/test_foo.py`.

## LangChain / AI Specifics
- Always set `temperature` explicitly when creating LLM instances — don't rely on defaults.
- Log or trace the full prompt sent to the LLM during development for debuggability.
- Keep prompt templates in `rag/prompts/`, registered as versioned `PromptVersion`s — never inline in business logic.
- When building chains, prefer LCEL (LangChain Expression Language) over legacy `Chain` classes.

