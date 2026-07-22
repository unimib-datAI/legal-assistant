import logging
from pathlib import Path

from legal_assistant import config
from legal_assistant.scraper.browser_fetcher import BrowserFetcher

logger = logging.getLogger(__name__)

"""
Utility class for general document handling tasks related to EUR-Lex.
"""


class EurlexDocumentUtils:
    HTML_DOCUMENT_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:"
    DOCUMENT_METADATA_URL = "https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:"

    def __init__(self, fetcher: BrowserFetcher | None = None):
        self.fetcher = fetcher or BrowserFetcher()
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

    def download_act_document(self, celex: str, output_dir: Path | str | None = None) -> Path:
        """Download an act document from EUR-Lex and save it as HTML.

        Defaults to ``config.CORPUS_DIR``, which is anchored to the repo root, so the file
        lands in the same place regardless of the working directory the CLI was run from.
        Returns the path to the saved file.
        """
        url = f"{self.HTML_DOCUMENT_URL}{celex}"
        output_path = Path(output_dir) if output_dir else config.CORPUS_DIR
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / f"{celex}.html"

        html = self.fetcher.fetch(url)
        file_path.write_text(html, encoding="utf-8")
        logger.info("Saved %s → %s (%d chars)", celex, file_path, len(html))

        return file_path
