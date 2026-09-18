"""Read the pages that have no text layer.

Two papers in `knowledge source/` are pure scans — 20 pages of images with not
one extractable character — so they contributed nothing to the knowledge base
and the consultant cannot discuss what they cover. `pypdf` reads a text layer;
a scan has none. These pages have to be rendered and recognised instead.

Why this combination:

- **pypdfium2** renders a PDF page to an image with no external program. The
  obvious alternative, `ocrmypdf`, needs Ghostscript and Tesseract installed
  system-wide, and produces a whole new PDF when all we want is the text.
  PyMuPDF renders just as well but is AGPL, which would reach into this
  repository.
- **Tesseract**, through `pytesseract`, does the recognition. It is the only
  part that is not a Python package, and the one thing a machine may lack.

Both are optional: they are needed to rebuild the knowledge base, never to run
HydroAssist, so they live in `requirements-ingest.txt` and are imported inside
the functions rather than at module import.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# 300 dpi is the resolution Tesseract's own documentation recommends for body
# text. Below about 200 it starts losing subscripts, which in these papers are
# the difference between S and S'.
DEFAULT_DPI = 300

# pypdfium2 renders in multiples of 72 dpi.
_BASE_DPI = 72.0


class OCRUnavailable(RuntimeError):
    """Raised when a page needs OCR and the tooling to do it is missing."""


@dataclass(frozen=True)
class OCRTooling:
    """What is installed, and what to do about what is not."""
    pypdfium2: bool
    pytesseract: bool
    tesseract_binary: Optional[str]

    @property
    def available(self) -> bool:
        return self.pypdfium2 and self.pytesseract and bool(self.tesseract_binary)

    @property
    def instructions(self) -> str:
        missing = []
        if not self.pypdfium2:
            missing.append("pypdfium2 (pip install -r requirements-ingest.txt)")
        if not self.pytesseract:
            missing.append("pytesseract (pip install -r requirements-ingest.txt)")
        if not self.tesseract_binary:
            missing.append(
                "the Tesseract program itself — https://github.com/UB-Mannheim/tesseract/wiki "
                "on Windows, `apt install tesseract-ocr` on Debian or Ubuntu, "
                "`brew install tesseract` on macOS")
        return "OCR needs: " + "; ".join(missing) if missing else "OCR is available."


def check_tooling() -> OCRTooling:
    """What is installed on this machine. Never raises, so a caller can report
    the whole picture instead of the first missing piece."""
    try:
        import pypdfium2  # noqa: F401
        has_pdfium = True
    except ImportError:
        has_pdfium = False

    binary = None
    try:
        import pytesseract
        has_pytesseract = True
        try:
            pytesseract.get_tesseract_version()
            binary = str(pytesseract.pytesseract.tesseract_cmd)
        except Exception:
            binary = None
    except ImportError:
        has_pytesseract = False

    return OCRTooling(has_pdfium, has_pytesseract, binary)


def ocr_available() -> bool:
    return check_tooling().available


def ocr_pdf_pages(pdf_path: str, page_numbers: List[int],
                  dpi: int = DEFAULT_DPI, language: str = "eng") -> Dict[int, str]:
    """Recognise text on the given 1-indexed pages.

    Only the pages asked for are rendered: OCR costs roughly a second a page, so
    a 300-page book whose text layer is fine must not pay for it.

    Raises OCRUnavailable, with what to install, if the tooling is missing.
    """
    tooling = check_tooling()
    if not tooling.available:
        raise OCRUnavailable(tooling.instructions)

    import pypdfium2
    import pytesseract

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    scale = dpi / _BASE_DPI
    recognised: Dict[int, str] = {}

    document = pypdfium2.PdfDocument(str(path))
    try:
        total = len(document)
        for page_number in page_numbers:
            if not 1 <= page_number <= total:
                logger.warning(f"{path.name}: no page {page_number} to OCR "
                               f"(it has {total})")
                continue
            image = document[page_number - 1].render(scale=scale).to_pil()
            try:
                text = pytesseract.image_to_string(image, lang=language)
            finally:
                image.close()
            recognised[page_number] = text.strip()
            logger.info(f"{path.name} page {page_number}: OCR read "
                        f"{len(recognised[page_number])} characters")
    finally:
        document.close()

    return recognised
