"""Main ingestion pipeline orchestrator."""

import json
from pathlib import Path
from typing import List, Dict, Optional
from langchain_core.documents import Document

from src.core.config import AppConfig
from src.ingestion.scrapers.usgs_scraper import download_usgs_documents
from src.ingestion.scrapers.aqtesolv_scraper import scrape_aqtesolv
from src.ingestion.processors.pdf_processor import PDFProcessor
from src.ingestion.processors.chunker import MetadataPreservingChunker
from src.ingestion.validators import validate_chunks_batch
from src.retrieval.vectorstore import VectorStoreManager
from src.retrieval.embeddings import embeddings_manager
from src.utils.logger import setup_logger
from src.utils.metadata import ChunkMetadata

logger = setup_logger(__name__)


class IngestionPipeline:
    """Orchestrates the complete data ingestion pipeline."""

    def __init__(self, config: AppConfig):
        """
        Initialize ingestion pipeline.

        Args:
            config: Application configuration
        """
        self.config = config
        self.pdf_processor = PDFProcessor()
        self.chunker = MetadataPreservingChunker(
            chunk_size=config.retrieval.chunk_size,
            chunk_overlap=config.retrieval.chunk_overlap
        )

        # Initialize embeddings and vector store
        self.embeddings = embeddings_manager.get_embeddings(
            config.retrieval.embedding_model
        )
        self.vectorstore = VectorStoreManager(
            persist_directory=config.vectorstore.persist_directory,
            collection_name=config.vectorstore.collection_name
        )

    def ingest(
        self,
        source: str,
        force_download: bool = False,
        validate: bool = True
    ) -> Dict[str, int]:
        """
        Run ingestion for a specific source.

        Args:
            source: Source name ('usgs', 'aqtesolv', or 'all')
            force_download: Force re-download of source data
            validate: Run validation on chunks

        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"Starting ingestion for source: {source}")

        stats = {"total_chunks": 0, "sources": [], "errors": 0}

        if source == "usgs" or source == "all":
            try:
                usgs_stats = self._ingest_usgs(force_download, validate)
                stats["total_chunks"] += usgs_stats["chunks"]
                stats["sources"].append("USGS_B1")
            except Exception as e:
                logger.error(f"USGS ingestion failed: {e}")
                stats["errors"] += 1

        if source == "aqtesolv" or source == "all":
            try:
                aq_stats = self._ingest_aqtesolv(validate)
                stats["total_chunks"] += aq_stats["chunks"]
                stats["sources"].append("AQTESOLV")
            except Exception as e:
                logger.error(f"AQTESOLV ingestion failed: {e}")
                stats["errors"] += 1

        logger.info(f"Ingestion complete: {stats}")

        return stats

    def _ingest_usgs(
        self,
        force_download: bool,
        validate: bool
    ) -> Dict[str, int]:
        """
        Ingest USGS documents.

        Args:
            force_download: Force re-download
            validate: Run validation

        Returns:
            Statistics dictionary
        """
        logger.info("Ingesting USGS TWI Book 3-B1...")

        # Download PDF
        pdf_path = download_usgs_documents(
            output_dir="./data/raw/usgs",
            force_download=force_download
        )

        if not pdf_path or not pdf_path.exists():
            logger.error("Failed to download USGS PDF")
            return {"chunks": 0}

        # Extract text from PDF
        logger.info("Extracting text from PDF...")
        pages = self.pdf_processor.extract_text_with_pages(str(pdf_path))

        # Process pages into documents
        documents = []

        for page_data in pages:
            page_num = page_data['page']
            text = page_data['text']
            has_equations = page_data['has_equations']

            if not text or len(text) < 100:
                logger.debug(f"Skipping page {page_num} (too short)")
                continue

            # Create metadata
            metadata = {
                'source': 'USGS_B1',
                'type': 'theory',  # Default, could be improved with classification
                'aquifer': 'general',  # Default, could be improved with classification
                'method': 'multiple',  # Multiple methods in document
                'page': page_num,
                'section': f'Page {page_num}',
                'doc_id': f'USGS_B1_page_{page_num}',
                'has_equations': has_equations,
                'confidence': 1.0
            }

            documents.append({
                'text': text,
                'metadata': metadata,
                'doc_id': f'USGS_B1_page_{page_num}'
            })

        logger.info(f"Prepared {len(documents)} documents from {len(pages)} pages")

        # Chunk documents
        all_chunks = self.chunker.chunk_documents(documents)

        logger.info(f"Created {len(all_chunks)} chunks")

        # Validate if requested
        if validate and self.config.ingestion.validate_metadata:
            chunk_dicts = [
                {
                    'content': chunk.page_content,
                    'metadata': chunk.metadata,
                    'doc_id': chunk.metadata['doc_id']
                }
                for chunk in all_chunks
            ]

            valid_count, errors = validate_chunks_batch(chunk_dicts)
            logger.info(f"Validation: {valid_count}/{len(all_chunks)} chunks valid")

            if errors:
                logger.warning(f"Validation errors: {errors[:5]}")  # Show first 5

        # Add to vector store
        self._add_chunks_to_vectorstore(all_chunks)

        return {"chunks": len(all_chunks)}

    def _ingest_aqtesolv(self, validate: bool) -> Dict[str, int]:
        """
        Ingest AQTESOLV method metadata.

        Args:
            validate: Run validation

        Returns:
            Statistics dictionary
        """
        logger.info("Ingesting AQTESOLV method metadata...")

        # Scrape (actually loads manually curated data)
        methods = scrape_aqtesolv(output_dir="./data/raw/aqtesolv")

        if not methods:
            logger.warning("No AQTESOLV methods loaded")
            return {"chunks": 0}

        # Convert methods to documents
        documents = []

        for method in methods:
            method_name = method['method']
            aquifer_type = method['aquifer']
            assumptions = method.get('assumptions', [])
            source_ref = method.get('source', 'Unknown')

            # Create text content
            text = f"Method: {method_name}\n\n"
            text += f"Aquifer Type: {aquifer_type}\n\n"
            text += f"Key Assumptions:\n"
            for assumption in assumptions:
                text += f"- {assumption}\n"
            text += f"\nSource: {source_ref}"

            # Create metadata (ChromaDB doesn't accept None values)
            metadata = {
                'source': 'AQTESOLV',
                'type': 'method_description',
                'aquifer': aquifer_type,
                'method': method_name,
                'page': 0,  # Changed from None to 0 (ChromaDB requirement)
                'section': f'{method_name} Method',
                'doc_id': f'AQTESOLV_{method_name}',
                'has_equations': False,
                'confidence': 1.0
            }

            documents.append({
                'text': text,
                'metadata': metadata,
                'doc_id': f'AQTESOLV_{method_name}'
            })

        logger.info(f"Prepared {len(documents)} method documents")

        # Chunk documents (though these are already small)
        all_chunks = self.chunker.chunk_documents(documents)

        logger.info(f"Created {len(all_chunks)} chunks")

        # Add to vector store
        self._add_chunks_to_vectorstore(all_chunks)

        return {"chunks": len(all_chunks)}

    def _add_chunks_to_vectorstore(self, chunks: List[Document]) -> None:
        """
        Add chunks to vector store.

        Args:
            chunks: List of LangChain Document objects
        """
        if not chunks:
            logger.warning("No chunks to add to vector store")
            return

        logger.info(f"Adding {len(chunks)} chunks to vector store...")

        # Prepare data for ChromaDB
        documents = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [chunk.metadata['chunk_id'] for chunk in chunks]

        # Add to vector store (ChromaDB will handle embedding internally if configured)
        # However, we need to add with our embeddings
        try:
            self.vectorstore.add_documents(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info("✓ Successfully added chunks to vector store")

        except Exception as e:
            logger.error(f"Failed to add chunks to vector store: {e}")
            raise

    def get_stats(self) -> Dict:
        """
        Get statistics about ingested data.

        Returns:
            Statistics dictionary
        """
        return self.vectorstore.get_collection_stats()

    def add_document(self, doc_data: Dict) -> None:
        """
        Add a single document to the vector store (for testing).

        Args:
            doc_data: Dictionary with 'content', 'metadata' keys
        """
        content = doc_data['content']
        metadata = doc_data['metadata']
        doc_id = metadata.get('chunk_id', 'test_doc')

        self.vectorstore.add_documents(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )

        logger.info(f"Added test document: {doc_id}")
