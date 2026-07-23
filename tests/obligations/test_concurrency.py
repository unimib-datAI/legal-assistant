"""Run a function over many items concurrently, retrying rate-limit errors with backoff.

The obligation stages make hundreds of independent LLM calls. Bounded concurrency is the
primary defence against a 429: only ``max_workers`` are ever in flight. Retry with
exponential backoff is the second, for the ones that slip through. Order is preserved so the
caller's per-passage numbering stays deterministic.
"""
from __future__ import annotations

import threading

import pytest

from legal_assistant.obligations.concurrency import map_concurrent


class FakeRateLimitError(Exception):
    """Stands in for openai.RateLimitError: detected by class name, no dependency."""


def test_results_keep_input_order():
    assert map_concurrent(lambda x: x * 2, [1, 2, 3, 4], max_workers=4) == [2, 4, 6, 8]


def test_an_empty_input_returns_empty():
    assert map_concurrent(lambda x: x, [], max_workers=4) == []


def test_a_rate_limit_error_is_retried_then_succeeds():
    calls = {"n": 0}

    def flaky(_):
        calls["n"] += 1
        if calls["n"] < 3:
            raise FakeRateLimitError("429")
        return "ok"

    result = map_concurrent(flaky, [1], max_workers=1, max_retries=5, base_delay=0)
    assert result == ["ok"]
    assert calls["n"] == 3


def test_a_non_retryable_error_is_raised_immediately():
    def boom(_):
        raise ValueError("not a rate limit")

    with pytest.raises(ValueError):
        map_concurrent(boom, [1], max_workers=1, base_delay=0)


def test_retries_are_bounded():
    calls = {"n": 0}

    def always(_):
        calls["n"] += 1
        raise FakeRateLimitError("429")

    with pytest.raises(FakeRateLimitError):
        map_concurrent(always, [1], max_workers=1, max_retries=3, base_delay=0)
    assert calls["n"] == 4  # first attempt + 3 retries


def test_concurrency_is_bounded_by_max_workers():
    """No more than max_workers tasks run at once, which is what keeps 429s away."""
    in_flight = {"now": 0, "peak": 0}
    lock = threading.Lock()
    gate = threading.Barrier(2, timeout=5)

    def track(_):
        with lock:
            in_flight["now"] += 1
            in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
        try:
            gate.wait()  # two tasks must overlap to trip the barrier
        except threading.BrokenBarrierError:
            pass
        with lock:
            in_flight["now"] -= 1
        return None

    map_concurrent(track, list(range(6)), max_workers=2, base_delay=0)
    assert in_flight["peak"] <= 2
