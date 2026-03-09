import logging
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

logger = logging.getLogger(__name__)

"""
Utility class for general document handling tasks related to EUR-Lex.
"""


class EurlexDocumentUtils:
    HTML_DOCUMENT_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:"
    DOCUMENT_METADATA_URL = "https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:"

    def __init__(self):
        self.document_config: list[dict] = []

    def build_document_config(self, celex: str) -> dict:
        """Build a document configuration dictionary for a given CELEX number."""
        file_path = self.download_act_document(celex)

        config = {
            "html_file": file_path,
            "celex": celex,
            "eurolex_url": f"{self.HTML_DOCUMENT_URL}{celex}",
            "document_info_url": f"{self.DOCUMENT_METADATA_URL}{celex}",
        }
        self.document_config.append(config)
        return config

    def download_act_document(
        self,
        celex: str,
        output_dir: str = "docs",
        wait_until: str = "networkidle",
        timeout_ms: int = 60_000,
    ) -> Path:
        """Download an act document from EUR-Lex using a headless browser.

        Uses Playwright/Chromium so that AWS WAF JavaScript challenges are
        solved transparently before the page content is captured.

        Returns the path to the saved HTML file.
        """
        url = f"{self.HTML_DOCUMENT_URL}{celex}"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / f"{celex}.html"

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()

            try:
                logger.info("Navigating to %s", url)
                page.goto(url, wait_until=wait_until, timeout=timeout_ms)
                html = page.content()
            except PlaywrightTimeoutError as exc:
                browser.close()
                raise TimeoutError(
                    f"Timed out waiting for EUR-Lex page for {celex}"
                ) from exc
            finally:
                browser.close()

        if len(html) < 1_000:
            raise ValueError(
                f"EUR-Lex returned a suspiciously short response for {celex} "
                f"({len(html)} chars) — possible bot-protection page"
            )

        file_path.write_text(html, encoding="utf-8")
        logger.info("Saved %s → %s (%d chars)", celex, file_path, len(html))

        return file_path
