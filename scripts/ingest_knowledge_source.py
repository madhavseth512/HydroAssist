#!/usr/bin/env python3
"""
Ingest all PDFs from knowledge source directory with proper aquifer type metadata.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import load_config
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.processors.pdf_processor import PDFProcessor
from src.ingestion.processors.chunker import MetadataPreservingChunker
from src.retrieval.vectorstore import VectorStoreManager
from src.retrieval.embeddings import embeddings_manager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Map folder names to aquifer types
AQUIFER_TYPE_MAP = {
    "Confined Aquifer": "confined",
    "Unconfined Aquifer": "unconfined",
    "Leaky Confined Aquifer": "leaky",
    "Fractured Aquifers": "fractured",
    "General": "general",
    "Step drawdown test": "general"
}


def ingest_knowledge_source():
    """Ingest all PDFs from knowledge source directory."""

    print("=" * 70)
    print("HydroAssist - Knowledge Source Ingestion")
    print("=" * 70)

    # Load config
    config = load_config()

    # Initialize components
    pdf_processor = PDFProcessor()
    chunker = MetadataPreservingChunker(
        chunk_size=config.retrieval.chunk_size,
        chunk_overlap=config.retrieval.chunk_overlap
    )
    embeddings = embeddings_manager.get_embeddings(config.retrieval.embedding_model)
    vectorstore = VectorStoreManager(
        persist_directory=config.vectorstore.persist_directory,
        collection_name=config.vectorstore.collection_name
    )

    # Find all PDFs
    knowledge_source_dir = Path("./knowledge source")

    if not knowledge_source_dir.exists():
        logger.error(f"Knowledge source directory not found: {knowledge_source_dir}")
        return

    # Process each subdirectory
    total_pdfs = 0
    total_chunks = 0

    for subdir in knowledge_source_dir.iterdir():
        if not subdir.is_dir():
            continue

        aquifer_type = AQUIFER_TYPE_MAP.get(subdir.name, "general")
        pdf_files = list(subdir.glob("*.pdf"))

        if not pdf_files:
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {subdir.name} ({len(pdf_files)} PDFs)")
        logger.info(f"Aquifer type: {aquifer_type}")
        logger.info(f"{'='*70}\n")

        for i, pdf_path in enumerate(pdf_files, 1):
            try:
                logger.info(f"[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")

                # Extract text from PDF
                pages = pdf_processor.extract_text_with_pages(str(pdf_path))

                if not pages:
                    logger.warning(f"  No pages extracted from {pdf_path.name}")
                    continue

                # Create documents
                documents = []
                for page_data in pages:
                    page_num = page_data['page']
                    text = page_data['text']
                    has_equations = page_data['has_equations']

                    if not text or len(text) < 100:
                        continue

                    # Create metadata
                    metadata = {
                        'source': f'Custom_{subdir.name}',
                        'type': 'research_paper',
                        'aquifer': aquifer_type,
                        'method': 'multiple',
                        'page': page_num,
                        'section': f'{pdf_path.stem} - Page {page_num}',
                        'doc_id': f'{pdf_path.stem}_page_{page_num}',
                        'has_equations': has_equations,
                        'confidence': 1.0
                    }

                    documents.append({
                        'text': text,
                        'metadata': metadata,
                        'doc_id': f'{pdf_path.stem}_page_{page_num}'
                    })

                # Chunk documents
                chunks = chunker.chunk_documents(documents)

                if not chunks:
                    logger.warning(f"  No chunks created from {pdf_path.name}")
                    continue

                # Add to vector store
                chunk_documents = [chunk.page_content for chunk in chunks]
                chunk_metadatas = [chunk.metadata for chunk in chunks]
                chunk_ids = [chunk.metadata['chunk_id'] for chunk in chunks]

                vectorstore.add_documents(
                    documents=chunk_documents,
                    metadatas=chunk_metadatas,
                    ids=chunk_ids
                )

                total_pdfs += 1
                total_chunks += len(chunks)

                logger.info(f"  ✓ Added {len(chunks)} chunks from {len(documents)} pages")

            except Exception as e:
                logger.error(f"  ✗ Failed to process {pdf_path.name}: {e}")
                continue

    # Print summary
    print("\n" + "=" * 70)
    print("Ingestion Complete!")
    print("=" * 70)
    print(f"Total PDFs processed: {total_pdfs}")
    print(f"Total chunks created: {total_chunks}")

    # Print collection stats
    stats = vectorstore.get_collection_stats()
    print(f"\nCollection Statistics:")
    print(f"  Total chunks in DB: {stats['total_chunks']}")
    print(f"  Sources:")
    for source, count in stats.get('sources', {}).items():
        print(f"    - {source}: {count} chunks")
    print(f"  Aquifer types:")
    for aquifer, count in stats.get('aquifer_types', {}).items():
        print(f"    - {aquifer}: {count} chunks")
    print("=" * 70)


if __name__ == "__main__":
    try:
        ingest_knowledge_source()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
