import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

"""
Utility class for general document handling tasks related to EUR-Lex.
"""
class EurlexDocumentUtils:
    HTML_DOCUMENT_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:"
    DOCUMENT_METADATA_URL = "https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:"

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()
        self.document_config: list[dict] = []

    def build_document_config(self, celex: str) -> dict:
        """Build a document configuration dictionary for a given CELEX number."""
        file_path = self.download_act_document(celex)

        config = {
            "html_file": file_path,
            "celex": celex,
            "eurolex_url": f"{self.HTML_DOCUMENT_URL}{celex}",
            "document_info_url": f"{self.DOCUMENT_METADATA_URL}{celex}"
        }
        self.document_config.append(config)
        return config

    def download_act_document(self, celex: str, output_dir: str = "docs") -> Path:
        """Download an act document from EUR-Lex and save it as HTML.

        Returns the path to the saved file.
        """
        url = f"{self.HTML_DOCUMENT_URL}{celex}"
        response = self.session.get(url)
        response.raise_for_status()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / f"{celex}.html"

        file_path.write_text(response.text, encoding="utf-8")
        logger.info("Saved %s to %s", celex, file_path)

        return file_path

