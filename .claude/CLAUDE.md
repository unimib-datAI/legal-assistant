# Legal Assistant Project

A Graph-RAG system over EU digital-regulation legislation. It models a knowledge graph from four acts — **AI Act**, **Data Governance Act**, **Data Act**, and **GDPR** — and exposes a retrieval-augmented generation (RAG) interface that answers legal questions grounded in those documents.

Queries are enriched by graph-aware topic filtering (ASKE) before being passed to an LLM, so answers are both semantically relevant and traceable to specific articles and paragraphs.

# Tech Stack

| Layer | Technology | Role |
|---|---|---|
| Language | Python | Entire codebase |
| Graph DB | Neo4j (Docker) | Stores nodes (Act, Chapter, Article, Paragraph, Concept) and vector embeddings |
| LLM orchestration | LangChain + `langchain-neo4j` | RetrievalQA chain, Neo4j vector store integration |
| LLM / Embeddings | OpenAI API (`langchain-openai`) | Answer generation and paragraph embeddings |
| Semantic re-ranking | `sentence-transformers` (`all-MiniLM-L6-v2`) | Topic similarity filtering in GraphEnrichedRetriever |
| NLP | NLTK | Tokenization and lemmatization in the ASKE pipeline |
| HTML parsing | BeautifulSoup4 | Scraping EUR-Lex legal documents from `docs/` |
| Config | `python-dotenv` | Loads `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `OPENAI_API_KEY` from `.env` |

# Architecture & Data Flow

The system has three sequential phases:

## Phase 1 — Graph Initialization (`graph_init.py`)
1. Reads the four legal HTML documents from `docs/` (GDPR, AI Act, Data Act, Data Governance Act)
2. `service/scraper/eurlex_exporter.py` parses each document into structured nodes: **Act → Chapter → Section → Article → Paragraph**
3. Cross-document citations and case-law relationships are extracted by `service/scraper/metadata_parser.py`
4. All nodes and relationships are written into Neo4j via `service/graph/graph_loader.py`

## Phase 2 — Topic Extraction (`aske_pipeline.py`)
1. Fetches all Paragraph nodes from Neo4j
2. Each paragraph is tokenized and lemmatized (NLTK)
3. `service/graph/aske.py` classifies tokens against seed terms in `service/graph/seed.py` (150+ legal concepts)
4. New concepts are clustered via affinity propagation (`service/topic/concept.py`) and written back to Neo4j as **Concept** nodes linked to paragraphs
5. Tunable parameters: `N_GENERATIONS`, `ALPHA`, `BETA`, `GAMMA`

## Phase 3 — RAG Query (`rag_pipeline.py`)
1. User question → OpenAI embeddings → Neo4j vector search retrieves candidate Paragraph nodes
2. `service/rag/rag_naive_with_topics.py` (`GraphEnrichedRetriever`) re-ranks candidates using SentenceTransformer (`all-MiniLM-L6-v2`) topic similarity (threshold: 0.35)
3. Top-k filtered paragraphs are passed as context to an OpenAI LLM via a LangChain `RetrievalQA` chain
4. Prompt templates live in `service/rag/prompt.py`

# Python Coding Rules

## Style & Structure
- Follow PEP 8. Use `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep functions short and single-purpose. If a function does more than one thing, split it.
- Prefer flat over nested — early returns instead of deeply nested `if/else`.
- Use type hints on all function signatures. Avoid `Any` unless truly necessary.
- Use `dataclasses` or `pydantic.BaseModel` for structured data instead of raw dicts.

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
- Keep test files mirroring source structure: `service/rag/foo.py` → `tests/service/rag/test_foo.py`.

## LangChain / AI Specifics
- Always set `temperature` explicitly when creating LLM instances — don't rely on defaults.
- Log or trace the full prompt sent to the LLM during development for debuggability.
- Keep prompt templates in dedicated files (`prompt.py`), not inline in business logic.
- When building chains, prefer LCEL (LangChain Expression Language) over legacy `Chain` classes.

