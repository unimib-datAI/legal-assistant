import logging

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

logger = logging.getLogger(__name__)

MIN_CONTENT_LENGTH = 1_000


class BrowserFetcher:
    """Fetches web pages using a headless Chromium browser.

    Solves JavaScript-based bot-protection challenges (e.g. AWS WAF)
    transparently by waiting for network activity to settle before
    capturing the rendered HTML.
    """

    def __init__(self, timeout_ms: int = 60_000, wait_until: str = "networkidle"):
        self.timeout_ms = timeout_ms
        self.wait_until = wait_until

    def fetch(self, url: str) -> str:
        """Navigate to *url* and return the fully-rendered HTML.

        Raises:
            TimeoutError: if the page does not finish loading within ``timeout_ms``.
            ValueError: if the returned HTML is shorter than ``MIN_CONTENT_LENGTH``
                chars, which indicates a bot-protection or error page.
        """
        logger.info("Fetching %s", url)

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                page.goto(url, wait_until=self.wait_until, timeout=self.timeout_ms)
                html = page.content()
            except PlaywrightTimeoutError as exc:
                raise TimeoutError(f"Timed out loading {url}") from exc
            finally:
                browser.close()

        if len(html) < MIN_CONTENT_LENGTH:
            raise ValueError(
                f"Suspiciously short response from {url} ({len(html)} chars) — "
                "possible bot-protection page"
            )

        logger.debug("Fetched %s (%d chars)", url, len(html))
        return html
