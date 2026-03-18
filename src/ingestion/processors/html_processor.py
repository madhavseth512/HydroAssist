"""HTML processing for web-scraped content."""

import re
from typing import Dict, List
from bs4 import BeautifulSoup
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class HTMLProcessor:
    """Process HTML content extracted from web pages."""

    def __init__(self):
        """Initialize HTML processor."""
        pass

    def extract_text(
        self,
        html_content: str,
        clean: bool = True
    ) -> str:
        """
        Extract text from HTML.

        Args:
            html_content: HTML string
            clean: Whether to clean extracted text

        Returns:
            Extracted text
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()

            # Get text
            text = soup.get_text()

            if clean:
                text = self._clean_text(text)

            return text

        except Exception as e:
            logger.error(f"Failed to extract text from HTML: {e}")
            return ""

    def extract_structured_content(
        self,
        html_content: str
    ) -> List[Dict[str, str]]:
        """
        Extract structured content from HTML (sections, headings, paragraphs).

        Args:
            html_content: HTML string

        Returns:
            List of content blocks with metadata
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            content_blocks = []
            current_section = "Introduction"

            # Find all headings and content
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'ul', 'ol']):
                if element.name in ['h1', 'h2', 'h3', 'h4']:
                    # Update current section
                    current_section = element.get_text().strip()

                elif element.name == 'p':
                    # Extract paragraph
                    text = element.get_text().strip()
                    if text and len(text) > 20:  # Skip very short paragraphs
                        content_blocks.append({
                            'type': 'paragraph',
                            'section': current_section,
                            'text': text
                        })

                elif element.name in ['ul', 'ol']:
                    # Extract list items
                    items = [li.get_text().strip() for li in element.find_all('li')]
                    if items:
                        content_blocks.append({
                            'type': 'list',
                            'section': current_section,
                            'text': '\n'.join(f"- {item}" for item in items)
                        })

            logger.info(f"Extracted {len(content_blocks)} content blocks")

            return content_blocks

        except Exception as e:
            logger.error(f"Failed to extract structured content: {e}")
            return []

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Strip
        text = text.strip()

        return text


def process_html(html_content: str, extract_structured: bool = False) -> any:
    """
    Convenience function to process HTML.

    Args:
        html_content: HTML string
        extract_structured: Whether to extract structured content

    Returns:
        Extracted text or structured content blocks
    """
    processor = HTMLProcessor()

    if extract_structured:
        return processor.extract_structured_content(html_content)
    else:
        return processor.extract_text(html_content)
