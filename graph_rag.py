import json
import asyncio
import time
import spacy
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI

import config
import logging
from dotenv import load_dotenv
from service.rag.prompt import GRAPH_RAG_MICROSOFT_PROMPT_v4

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Rate limit config (tune to your OpenAI tier) ─────────────────────────────
MAX_CONCURRENT = 5        # lower concurrency → fewer simultaneous token bursts
TPM_LIMIT      = 180_000  # stay under 200k TPM with 10% safety margin
MIN_CHUNK_CHARS = 100

# ── "costs" removed — v4 prompt now processes it ─────────────────────────────
SKIP_SECTIONS = ["[signatures]", "table of contents"]

entity_types = [
    "COURT", "REFERRING_COURT", "NATIONAL_AUTHORITY", "EU_INSTITUTION",
    "APPLICANT", "DEFENDANT", "INTERVENER", "ADVOCATE_GENERAL",
    "REGULATION", "DIRECTIVE", "TREATY_PROVISION", "NATIONAL_LAW",
    "ARTICLE", "RECITAL", "LEGAL_PRINCIPLE", "PRELIMINARY_QUESTION",
    "LEGAL_ISSUE", "FACTUAL_CIRCUMSTANCE", "ADMINISTRATIVE_DECISION",
    "OPERATIVE_PART", "HOLDING",
]

# ── TPM-aware rate limiter ────────────────────────────────────────────────────
class TPMRateLimiter:
    """
    Sliding window token-per-minute limiter.
    Tracks when tokens were consumed and blocks until capacity is available.
    """
    def __init__(self, tpm_limit: int):
        self.tpm_limit = tpm_limit
        self.window    = 60.0
        self.timestamps: list[tuple[float, int]] = []
        self._lock     = asyncio.Lock()

    async def acquire(self, tokens: int):
        async with self._lock:
            while True:
                now = time.monotonic()
                self.timestamps = [
                    (t, tok) for t, tok in self.timestamps
                    if now - t < self.window
                ]
                used = sum(tok for _, tok in self.timestamps)

                if used + tokens <= self.tpm_limit:
                    self.timestamps.append((now, tokens))
                    return

                oldest_time = self.timestamps[0][0]
                wait = self.window - (now - oldest_time) + 0.1
                logger.debug("TPM limit approaching, waiting %.2fs", wait)
                await asyncio.sleep(wait)

# ── Token estimation ──────────────────────────────────────────────────────────
def _estimate_tokens(prompt: str) -> int:
    """
    Reliable estimate: 1 token ≈ 0.75 words for English legal text.
    Count actual prompt words rather than using a fixed constant.
    Add 600 token buffer for the JSON output.
    """
    word_count = len(prompt.split())
    return int(word_count / 0.75) + 600

# ── Chunking ──────────────────────────────────────────────────────────────────
def _document_text_chunking(document_text: str, nlp) -> list[str]:
    doc = nlp(document_text)
    return [sent.text.strip() for sent in doc.sents]

def is_valid_chunk(chunk: str) -> bool:
    stripped = chunk.strip().lower()
    if len(stripped) < MIN_CHUNK_CHARS:
        return False
    if any(stripped.startswith(s) for s in SKIP_SECTIONS):
        return False
    return True

def filter_chunks(chunks: list[str]) -> list[str]:
    valid, skipped = [], []
    for chunk in chunks:
        (valid if is_valid_chunk(chunk) else skipped).append(chunk)
    if skipped:
        logger.info("Skipped %d/%d chunks", len(skipped), len(chunks))
    return valid

# ── Async extraction ──────────────────────────────────────────────────────────
async def _extract_chunk(
    client:       AsyncOpenAI,
    semaphore:    asyncio.Semaphore,
    rate_limiter: TPMRateLimiter,
    chunk:        str,
    chunk_id:     str,
    entity_types: list[str],
) -> dict:
    prompt = (
        GRAPH_RAG_MICROSOFT_PROMPT_v4
        .replace("{entity_types}", ", ".join(entity_types))
        .replace("{chunk_id}",    chunk_id)
        .replace("{input_text}",  chunk)
        .replace("{chunk_text}",  chunk)
    )

    estimated_tokens = _estimate_tokens(prompt)  # dynamic, based on actual prompt size

    async with semaphore:
        await rate_limiter.acquire(estimated_tokens)
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(response.choices[0].message.content)
            logger.debug("OK %s (~%d tokens)", chunk_id, estimated_tokens)
            return {
                "chunk_id":      chunk_id,
                "chunk_text":    chunk,
                "entities":      parsed.get("entities", []),
                "relationships": parsed.get("relationships", []),
            }

        except Exception as e:
            logger.error("Failed %s: %s", chunk_id, e)
            return {
                "chunk_id":      chunk_id,
                "chunk_text":    chunk,
                "entities":      [],
                "relationships": [],
            }

async def _extract_all(
    chunks:       list[str],
    case_id:      str,
    entity_types: list[str],
) -> list[dict]:
    client       = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    semaphore    = asyncio.Semaphore(MAX_CONCURRENT)
    rate_limiter = TPMRateLimiter(tpm_limit=TPM_LIMIT)

    tasks = [
        _extract_chunk(
            client, semaphore, rate_limiter,
            chunk,
            f"{case_id}_chunk_{idx:03d}",
            entity_types,
        )
        for idx, chunk in enumerate(chunks)
    ]

    results = await tqdm_asyncio.gather(*tasks, desc="Extracting elements")
    return sorted(results, key=lambda x: x["chunk_id"])

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    case_id     = "62019CJ0645"
    source_path = Path("results/case_law_result.txt")
    output_path = Path("results/v4/elements.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading spaCy...")
    nlp = spacy.load("en_core_web_sm")

    logger.info("Chunking document...")
    full_document = source_path.read_text(encoding="utf-8")
    raw_chunks    = _document_text_chunking(full_document, nlp)
    chunks        = filter_chunks(raw_chunks)

    # estimate using actual prompt size on first chunk
    sample_prompt = (
        GRAPH_RAG_MICROSOFT_PROMPT_v4
        .replace("{entity_types}", ", ".join(entity_types))
        .replace("{chunk_id}",    f"{case_id}_chunk_000")
        .replace("{input_text}",  chunks[0] if chunks else "")
        .replace("{chunk_text}",  chunks[0] if chunks else "")
    )
    sample_tokens = _estimate_tokens(sample_prompt)
    total_tokens  = sample_tokens * len(chunks)
    expected_mins = total_tokens / TPM_LIMIT

    logger.info(
        "Processing %d chunks (~%d tokens/chunk, ~%d total, estimated %.1f min at %d TPM)",
        len(chunks), sample_tokens, total_tokens, expected_mins, TPM_LIMIT
    )

    results = asyncio.run(_extract_all(chunks, case_id, entity_types))

    failed = [r for r in results if not r["entities"] and not r["relationships"]]
    if failed:
        logger.warning(
            "%d chunks failed — consider reprocessing: %s",
            len(failed), [r["chunk_id"] for r in failed]
        )

    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved %d chunks to %s", len(results), output_path)

if __name__ == "__main__":
    main()