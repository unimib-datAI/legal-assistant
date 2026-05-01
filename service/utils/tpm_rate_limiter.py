import asyncio
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class TPMRateLimiter:
    """
    Sliding window token-per-minute limiter.
    Tracks when tokens were consumed and blocks until capacity is available.
    """

    def __init__(self, tpm_limit: int):
        self.tpm_limit = tpm_limit
        self.window = 60.0
        self.timestamps: list[tuple[float, int]] = []
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()

                # Drop entries that have fallen outside the sliding window
                active_entries = []
                for timestamp, token_count in self.timestamps:
                    if now - timestamp < self.window:
                        active_entries.append((timestamp, token_count))
                self.timestamps = active_entries

                # Count how many tokens have been consumed in the current window
                consumed_tokens = 0
                for _, token_count in self.timestamps:
                    consumed_tokens += token_count

                # If there is enough capacity, record the usage and proceed
                if consumed_tokens + tokens <= self.tpm_limit:
                    self.timestamps.append((now, tokens))
                    return

                # Otherwise wait until the oldest entry expires and try again
                oldest_timestamp = self.timestamps[0][0]
                time_since_oldest = now - oldest_timestamp
                time_until_oldest_expires = self.window - time_since_oldest
                wait = time_until_oldest_expires + 0.1

                logger.debug("TPM limit approaching, waiting %.2fs", wait)
                await asyncio.sleep(wait)
