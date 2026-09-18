"""PDF text extraction with page-level metadata."""

import re
from pathlib import Path
from typing import List, Dict, Optional
from pypdf import PdfReader
from src.ingestion.processors.ocr import DEFAULT_DPI, OCRUnavailable, ocr_pdf_pages
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# A page holding less than this has no usable text layer: it is a scan, a figure
# or a title page. Ingestion drops such pages, so it is also the point at which
# OCR is worth its second a page.
MIN_PAGE_CHARS = 100

# How a document's pages may be read.
OCR_NEVER = "never"     # text layer only — what every caller did before OCR existed
OCR_AUTO = "auto"       # OCR only the pages the text layer failed on
OCR_ALWAYS = "always"   # OCR every page, ignoring the text layer
OCR_MODES = (OCR_NEVER, OCR_AUTO, OCR_ALWAYS)


class PDFProcessor:
    """Extract text from PDFs with metadata preservation."""

    def __init__(self):
        """Initialize PDF processor."""
        self.logger = setup_logger(__name__)

    def extract_text_with_pages(
        self,
        pdf_path: str,
        clean_text: bool = True,
        ocr: str = OCR_NEVER,
        min_chars: int = MIN_PAGE_CHARS,
        dpi: int = DEFAULT_DPI,
    ) -> List[Dict[str, any]]:
        """
        Extract text from PDF with page-level metadata.

        Args:
            pdf_path: Path to PDF file
            clean_text: Whether to clean extracted text
            ocr: OCR_NEVER (default), OCR_AUTO to recognise only the pages whose
                text layer yielded less than `min_chars`, or OCR_ALWAYS.
                OCR_AUTO is what the corpus scripts use: two papers in
                `knowledge source/` are pure scans and several more carry a text
                layer thin enough to be useless.
            min_chars: The yield below which a page counts as having no text.
            dpi: Rendering resolution for OCR.

        Returns:
            List of dicts with 'page', 'text', 'has_equations' and 'ocr' keys,
            the last saying whether the text was recognised rather than read.

        Raises:
            OCRUnavailable: if OCR is asked for and the tooling is missing.
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if ocr not in OCR_MODES:
            raise ValueError(f"Unknown ocr mode {ocr!r}; expected one of "
                             f"{', '.join(OCR_MODES)}.")

        logger.info(f"Processing PDF: {pdf_path.name}")

        try:
            reader = PdfReader(str(pdf_path))
            num_pages = len(reader.pages)

            logger.info(f"PDF has {num_pages} pages")

            pages_data = []

            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text = page.extract_text()

                if clean_text:
                    text = self._clean_text(text)

                # Detect equations (simple heuristic)
                has_equations = self._detect_equations(text)

                pages_data.append({
                    'page': page_num + 1,  # 1-indexed
                    'text': text,
                    'has_equations': has_equations,
                    'ocr': False
                })

            if ocr != OCR_NEVER:
                self._recognise_pages(pdf_path, pages_data, ocr, min_chars, dpi,
                                      clean_text)

            logger.info(f"Successfully extracted text from {len(pages_data)} pages")

            return pages_data

        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            raise

    def _recognise_pages(
        self,
        pdf_path: Path,
        pages_data: List[Dict],
        ocr: str,
        min_chars: int,
        dpi: int,
        clean_text: bool,
    ) -> None:
        """Fill in the pages the text layer could not supply, in place.

        A recognised page replaces the extracted one only when it yields more
        text: OCR of a page that was already readable is usually worse than the
        text layer, and on a blank page it returns noise.
        """
        wanted = [page['page'] for page in pages_data
                  if ocr == OCR_ALWAYS or len(page['text'] or "") < min_chars]

        if not wanted:
            return

        logger.info(f"{pdf_path.name}: OCR for {len(wanted)} of "
                    f"{len(pages_data)} pages")

        recognised = ocr_pdf_pages(str(pdf_path), wanted, dpi=dpi)

        by_number = {page['page']: page for page in pages_data}
        for page_number, text in recognised.items():
            page = by_number[page_number]
            if clean_text:
                text = self._clean_text(text)
            if len(text) <= len(page['text'] or ""):
                continue
            page['text'] = text
            page['has_equations'] = self._detect_equations(text)
            page['ocr'] = True

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove common header/footer patterns
        # Pattern: page numbers, running headers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

        # Remove form feed characters
        text = text.replace('\f', '\n')

        # Normalize line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def _detect_equations(self, text: str) -> bool:
        """
        Detect if text contains mathematical equations.

        Uses simple heuristics to detect mathematical notation.

        Args:
            text: Text to check

        Returns:
            True if equations likely present
        """
        # Check for common math symbols and patterns
        math_indicators = [
            r'[+\-*/=]{2,}',  # Multiple operators
            r'\b[a-zA-Z]\s*=\s*',  # Variable assignments
            r'[∫∑∏√π]',  # Math symbols
            r'\b(equation|formula|derivative|integral)\b',  # Keywords
            r'\([^)]*[+\-*/=][^)]*\)',  # Expressions in parentheses
        ]

        for pattern in math_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def extract_table_of_contents(
        self,
        pdf_path: str
    ) -> List[Dict[str, any]]:
        """
        Extract table of contents from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of TOC entries with title and page
        """
        pdf_path = Path(pdf_path)

        try:
            reader = PdfReader(str(pdf_path))

            # Try to extract outlines (bookmarks)
            outlines = reader.outline if hasattr(reader, 'outline') else []

            toc = []

            def extract_outline_items(items, level=0):
                """Recursively extract outline items."""
                for item in items:
                    if isinstance(item, dict):
                        title = item.get('/Title', 'Unknown')
                        page_num = None

                        # Try to get page number
                        if '/Page' in item:
                            page_obj = item['/Page']
                            if hasattr(reader, 'pages'):
                                try:
                                    page_num = reader.pages.index(page_obj) + 1
                                except:
                                    pass

                        toc.append({
                            'title': title,
                            'page': page_num,
                            'level': level
                        })
                    elif isinstance(item, list):
                        extract_outline_items(item, level + 1)

            if outlines:
                extract_outline_items(outlines)
                logger.info(f"Extracted {len(toc)} TOC entries")
            else:
                logger.warning("No table of contents found in PDF")

            return toc

        except Exception as e:
            logger.warning(f"Failed to extract TOC: {e}")
            return []


def process_pdf(pdf_path: str, clean_text: bool = True) -> List[Dict]:
    """
    Convenience function to process a PDF.

    Args:
        pdf_path: Path to PDF file
        clean_text: Whether to clean text

    Returns:
        List of page data dictionaries
    """
    processor = PDFProcessor()
    return processor.extract_text_with_pages(pdf_path, clean_text=clean_text)
