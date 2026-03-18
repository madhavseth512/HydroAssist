"""PDF text extraction with page-level metadata."""

import re
from pathlib import Path
from typing import List, Dict, Optional
from pypdf import PdfReader
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PDFProcessor:
    """Extract text from PDFs with metadata preservation."""

    def __init__(self):
        """Initialize PDF processor."""
        self.logger = setup_logger(__name__)

    def extract_text_with_pages(
        self,
        pdf_path: str,
        clean_text: bool = True
    ) -> List[Dict[str, any]]:
        """
        Extract text from PDF with page-level metadata.

        Args:
            pdf_path: Path to PDF file
            clean_text: Whether to clean extracted text

        Returns:
            List of dicts with 'page', 'text', and 'has_equations' keys
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

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
                    'has_equations': has_equations
                })

            logger.info(f"Successfully extracted text from {len(pages_data)} pages")

            return pages_data

        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            raise

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
