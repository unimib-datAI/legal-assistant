import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root (directory containing this config file)
PROJECT_ROOT = Path(__file__).resolve().parent

# Evaluation directories
EVALS_DIR = PROJECT_ROOT / "test" / "rag_eval" / "evals"

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

# Cross-encoder reranker (sentence-transformers CrossEncoder id). Overridable via env to
# A/B alternative rerankers with retrieval_eval.py, e.g. RERANK_MODEL=zeroentropy/zerank-2 - BAAI/bge-reranker-v2-m3.
RERANK_MODEL = os.getenv("RERANK_MODEL", "zeroentropy/zerank-2")
# Some rerankers (e.g. Jina) ship custom modeling code and need this to load via CrossEncoder.
RERANK_TRUST_REMOTE_CODE = os.getenv("RERANK_TRUST_REMOTE_CODE", "").lower() in ("1", "true", "yes")
# Torch dtype for the reranker weights. "auto" respects the checkpoint's native dtype, so
# large decoder rerankers (e.g. the ~4B bf16 zerank-2) load in bf16 instead of fp32 — ~half
# the VRAM and much faster on a 16GB GPU, where fp32 spills to CPU. Override with e.g. "float32".
RERANK_DTYPE = os.getenv("RERANK_DTYPE", "auto")