"""
USGS document downloader.

Downloads USGS Techniques of Water-Resources Investigations (TWI)
Book 3, Chapter B1: "Methods of Determining Permeability, Transmissibility, and Drawdown"

This is a PUBLIC DOMAIN document, freely available for download and use.
"""

import hashlib
import requests
from pathlib import Path
from typing import Optional
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class USGSScraper:
    """Downloader for USGS public domain documents."""

    # USGS TWI Book 3-B1 URLs
    PRIMARY_URL = "https://pubs.usgs.gov/twri/twri3-b1/"
    ALTERNATIVE_URL = "https://pubs.usgs.gov/publication/twri03B1"

    # Known PDF URLs (these are public domain)
    PDF_URLS = [
        "https://pubs.usgs.gov/publication/twri03B1/pdf",
        "https://pubs.usgs.gov/twri/twri3-b1/pdf/TWRI_3-B1.pdf",
        "https://pubs.usgs.gov/twri/twri3-b1/pdf/twri_3-B1.pdf",
        "https://pubs.er.usgs.gov/publication/twri03B1",
    ]

    DOCUMENT_NAME = "USGS_TWI_Book3_B1.pdf"

    def __init__(self, output_dir: str = "./data/raw/usgs"):
        """
        Initialize USGS scraper.

        Args:
            output_dir: Directory to save downloaded files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_twri_b1(
        self,
        force_download: bool = False
    ) -> Optional[Path]:
        """
        Download USGS TWI Book 3-B1 PDF.

        Args:
            force_download: Force re-download even if cached

        Returns:
            Path to downloaded PDF, or None if failed
        """
        output_path = self.output_dir / self.DOCUMENT_NAME

        # Check if already downloaded
        if output_path.exists() and not force_download:
            logger.info(f"PDF already exists: {output_path}")
            logger.info(f"Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
            return output_path

        logger.info("Downloading USGS TWI Book 3-B1...")

        # Try each URL until one works
        for url in self.PDF_URLS:
            logger.info(f"Trying URL: {url}")

            try:
                response = requests.get(url, timeout=30, stream=True)
                response.raise_for_status()

                # Check content type
                content_type = response.headers.get('Content-Type', '')
                if 'pdf' not in content_type.lower():
                    logger.warning(f"Unexpected content type: {content_type}")
                    continue

                # Download with progress
                total_size = int(response.headers.get('content-length', 0))
                logger.info(f"Downloading {total_size / (1024*1024):.2f} MB...")

                with open(output_path, 'wb') as f:
                    downloaded = 0
                    chunk_size = 8192

                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            # Log progress every 1MB
                            if downloaded % (1024 * 1024) < chunk_size:
                                progress = (downloaded / total_size * 100) if total_size > 0 else 0
                                logger.debug(f"Progress: {progress:.1f}%")

                # Verify download
                if not output_path.exists() or output_path.stat().st_size < 1000:
                    logger.error("Download failed or file too small")
                    output_path.unlink(missing_ok=True)
                    continue

                logger.info(f"✓ Successfully downloaded to: {output_path}")
                logger.info(f"Size: {output_path.stat().st_size / (1024*1024):.2f} MB")

                # Compute checksum
                checksum = self._compute_checksum(output_path)
                logger.info(f"SHA256: {checksum[:16]}...")

                return output_path

            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to download from {url}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error downloading from {url}: {e}")
                continue

        logger.error("All download attempts failed")
        return None

    def _compute_checksum(self, file_path: Path) -> str:
        """
        Compute SHA256 checksum of file.

        Args:
            file_path: Path to file

        Returns:
            Hexadecimal checksum string
        """
        sha256 = hashlib.sha256()

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)

        return sha256.hexdigest()

    def verify_download(
        self,
        file_path: Path,
        expected_checksum: Optional[str] = None
    ) -> bool:
        """
        Verify downloaded file integrity.

        Args:
            file_path: Path to file
            expected_checksum: Expected SHA256 checksum (if known)

        Returns:
            True if verification passed
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False

        # Check file size (should be at least 1MB)
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb < 1.0:
            logger.error(f"File too small: {size_mb:.2f} MB")
            return False

        # Compute checksum
        actual_checksum = self._compute_checksum(file_path)

        if expected_checksum:
            if actual_checksum == expected_checksum:
                logger.info("✓ Checksum verification passed")
                return True
            else:
                logger.error("✗ Checksum mismatch!")
                logger.error(f"Expected: {expected_checksum}")
                logger.error(f"Actual:   {actual_checksum}")
                return False
        else:
            logger.info(f"Checksum: {actual_checksum}")
            return True


def download_usgs_documents(
    output_dir: str = "./data/raw/usgs",
    force_download: bool = False
) -> Optional[Path]:
    """
    Convenience function to download USGS documents.

    Args:
        output_dir: Output directory
        force_download: Force re-download

    Returns:
        Path to downloaded PDF
    """
    scraper = USGSScraper(output_dir)
    return scraper.download_twri_b1(force_download=force_download)
