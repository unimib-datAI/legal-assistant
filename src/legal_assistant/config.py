import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Repository root: this file lives at <root>/src/legal_assistant/config.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Evaluation datasets and result CSVs (see evals/).
EVALS_DIR = PROJECT_ROOT / "evals" / "evals"

# Source data: the EUR-Lex act HTML the scraper downloads and the graph is built from.
# Anchored to the repo root so a run does not depend on the current working directory.
CORPUS_DIR = PROJECT_ROOT / "corpus"

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPEN_API_KEY")
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', '')
OPENAI_ENDPOINT = f"{OPENAI_BASE_URL}/embeddings" if OPENAI_BASE_URL else None

# Chat model used by the RAG pipeline (classifier, answer synthesis, answer filter).
# Overridable via env so synthesis can be bumped (e.g. to gpt-4o) without a code change.
RAG_LLM_MODEL = os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")

# Embedding model used both at graph-build time (to embed Paragraph/Recital/Article nodes) and
# at query time (to embed the user question against the Neo4j vector index). These two MUST use
# the same model, so keep it centralized here. EMBEDDING_DIM must match the model's output size
# (text-embedding-3-small → 1536) and drives the Neo4j vector index dimensionality.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

# Minimum per-act relevance (0-1) the query classifier must assign for an act to become a
# retrieval target. Lower selects more acts (higher recall, more off-topic context); higher
# is stricter. Tunable via env for threshold sweeps. See legal_assistant/rag/intent_classifier.py.
ACT_SCORE_THRESHOLD = float(os.getenv("ACT_SCORE_THRESHOLD", "0.4"))

# Cross-encoder reranker (sentence-transformers CrossEncoder id). Overridable via env to
# A/B alternative rerankers with retrieval_eval.py, e.g. RERANK_MODEL=zeroentropy/zerank-2 - BAAI/bge-reranker-v2-m3.
RERANK_MODEL = os.getenv("RERANK_MODEL", "zeroentropy/zerank-2")
# Some rerankers (e.g. Jina) ship custom modeling code and need this to load via CrossEncoder.
RERANK_TRUST_REMOTE_CODE = os.getenv("RERANK_TRUST_REMOTE_CODE", "").lower() in ("1", "true", "yes")
# Torch dtype for the reranker weights. "auto" respects the checkpoint's native dtype, so
# large decoder rerankers (e.g. the ~4B bf16 zerank-2) load in bf16 instead of fp32 — ~half
# the VRAM and much faster on a 16GB GPU, where fp32 spills to CPU. Override with e.g. "float32".
RERANK_DTYPE = os.getenv("RERANK_DTYPE", "auto")