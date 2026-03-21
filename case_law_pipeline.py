import logging
import tempfile
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from playwright.sync_api import sync_playwright

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

url = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:62019CJ0645"

converter = DocumentConverter(
    allowed_formats=[InputFormat.PDF],
    format_options={InputFormat.PDF: PdfFormatOption()},
)


def fetch_pdf(pdf_url: str) -> Path:
    """Download a PDF from *pdf_url* using a headless browser and return the local path."""
    logger.info("Downloading PDF from %s", pdf_url)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        with page.expect_download() as download_info:
            page.goto(pdf_url, wait_until="networkidle", timeout=60_000)

        download = download_info.value
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.close()
        download.save_as(tmp.name)
        browser.close()

    path = Path(tmp.name)
    logger.info("Saved PDF to %s (%d bytes)", path, path.stat().st_size)
    return path


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

pdf_path = fetch_pdf(url)
try:
    doc = converter.convert(pdf_path).document
finally:
    pdf_path.unlink(missing_ok=True)
    logger.info("Deleted temp file %s", pdf_path)

output_path = RESULTS_DIR / "case_law_result.md"
output_path.write_text(doc.export_to_markdown(), encoding="utf-8")
logger.info("Saved markdown to %s", output_path)