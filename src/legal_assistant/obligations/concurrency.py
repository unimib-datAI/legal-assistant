"""Run a function over many items concurrently, retrying rate-limit errors with backoff.

The obligation stages fire hundreds of independent LLM calls. Two defences against a provider
429 keep them from failing the batch:

* **Bounded concurrency.** At most ``max_workers`` calls are ever in flight, which is what
  keeps the request rate under the provider's ceiling in the first place.
* **Retry with exponential backoff.** A call that is rate-limited anyway waits, longer each
  time, with jitter so a burst of retries does not resynchronise into a new spike.

Order is preserved, so a caller relying on deterministic per-item numbering is unaffected by
the order tasks happen to finish in.
"""
from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def is_rate_limit(exc: BaseException) -> bool:
    """Whether an exception is a provider rate-limit, without importing the provider SDK.

    Matches by HTTP status, a numeric ``code``, or the exception's class name, so it catches
    ``openai.RateLimitError`` and its equivalents without a hard dependency on any of them.
    """
    if getattr(exc, "status_code", None) == 429 or getattr(exc, "code", None) == 429:
        return True
    return "ratelimit" in type(exc).__name__.lower()


def _with_retry(
    fn: Callable[[T], R],
    item: T,
    max_retries: int,
    base_delay: float,
    is_retryable: Callable[[BaseException], bool],
) -> R:
    """Call ``fn(item)``, retrying a retryable exception with exponential backoff and jitter."""
    for attempt in range(max_retries + 1):
        try:
            return fn(item)
        except BaseException as exc:  # noqa: BLE001 - re-raised unless retryable and budget left
            if attempt >= max_retries or not is_retryable(exc):
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
            logger.warning("[concurrency] rate-limited, retry %d/%d in %.1fs",
                           attempt + 1, max_retries, delay)
            if delay:
                time.sleep(delay)
    raise RuntimeError("unreachable")  # pragma: no cover


def map_concurrent(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    max_workers: int = 8,
    max_retries: int = 6,
    base_delay: float = 1.0,
    is_retryable: Callable[[BaseException], bool] = is_rate_limit,
) -> List[R]:
    """Map ``fn`` over ``items`` with bounded concurrency, in input order.

    A task that keeps hitting the retryable error past ``max_retries`` raises, failing the
    whole map: a persistent rate limit is a real problem, not something to swallow.
    """
    if not items:
        return []

    results: List[R] = [None] * len(items)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_with_retry, fn, item, max_retries, base_delay, is_retryable): index
            for index, item in enumerate(items)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results
