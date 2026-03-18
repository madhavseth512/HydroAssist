#!/usr/bin/env python3
"""
Ingest custom PDF documents into HydroAssist knowledge base.

This script processes user-provided hydrogeology PDFs and adds them
to the vector store with appropriate metadata.

Usage:
    python scripts/ingest_custom_pdfs.py --pdf-dir ./data/raw/custom_pdfs
    python scripts/ingest_custom_pdfs.py --pdf-file path/to/document.pdf
    python scripts/ingest_custom_pdfs.py --pdf-dir ./my_pdfs --source "Custom Source"
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.processors.pdf_processor import PDFProcessor
from src.ingestion.processors.chunker import MetadataPreservingChunker
from src.retrieval.vectorstore import VectorStoreManager
from src.retrieval.embeddings import embeddings_manager
from src.core.config import load_config
from src.utils.logger import setup_logger
from src.utils.metadata import create_chunk_id

logger = setup_logger("CustomPDFIngest")


def ingest_pdf_file(
    pdf_path: Path,
    config,
    vectorstore: VectorStoreManager,
    chunker: MetadataPreservingChunker,
    source_name: str = "Custom",
    aquifer_type: str = "general",
    method: str = "multiple"
) -> int:
    """
    Ingest a single PDF file.

    Args:
        pdf_path: Path to PDF file
        config: Application configuration
        vectorstore: Vector store manager
        chunker: Text chunker
        source_name: Source identifier
        aquifer_type: Aquifer type (confined, unconfined, leaky, fractured, general)
        method: Method name

    Returns:
        Number of chunks created
    """
    logger.info(f"Processing: {pdf_path.name}")

    # Extract text from PDF
    pdf_processor = PDFProcessor()
    try:
        pages = pdf_processor.extract_text_with_pages(str(pdf_path))
    except Exception as e:
        logger.error(f"Failed to process {pdf_path}: {e}")
        return 0

    # Process pages into documents
    documents = []
    doc_id_base = pdf_path.stem.replace(" ", "_").replace("-", "_")

    for page_data in pages:
        page_num = page_data['page']
        text = page_data['text']
        has_equations = page_data['has_equations']

        if not text or len(text) < 100:
            logger.debug(f"Skipping page {page_num} (too short)")
            continue

        # Create metadata
        metadata = {
            'source': source_name,
            'type': 'theory',  # Can be customized
            'aquifer': aquifer_type,
            'method': method,
            'page': page_num,
            'section': f'Page {page_num}',
            'doc_id': f'{doc_id_base}_page_{page_num}',
            'has_equations': has_equations,
            'confidence': 1.0,
            'filename': pdf_path.name
        }

        documents.append({
            'text': text,
            'metadata': metadata,
            'doc_id': f'{doc_id_base}_page_{page_num}'
        })

    logger.info(f"Prepared {len(documents)} documents from {len(pages)} pages")

    if not documents:
        logger.warning(f"No valid documents extracted from {pdf_path}")
        return 0

    # Chunk documents
    all_chunks = chunker.chunk_documents(documents)

    logger.info(f"Created {len(all_chunks)} chunks")

    # Add to vector store
    try:
        vectorstore.add_documents(
            documents=[chunk.page_content for chunk in all_chunks],
            metadatas=[chunk.metadata for chunk in all_chunks],
            ids=[chunk.metadata['chunk_id'] for chunk in all_chunks]
        )
        logger.info(f"✓ Added {len(all_chunks)} chunks to vector store")
    except Exception as e:
        logger.error(f"Failed to add chunks to vector store: {e}")
        return 0

    return len(all_chunks)


def ingest_pdf_directory(
    pdf_dir: Path,
    config,
    vectorstore: VectorStoreManager,
    chunker: MetadataPreservingChunker,
    source_name: str = "Custom",
    aquifer_type: str = "general",
    method: str = "multiple"
) -> int:
    """
    Ingest all PDF files from a directory.

    Args:
        pdf_dir: Directory containing PDF files
        config: Application configuration
        vectorstore: Vector store manager
        chunker: Text chunker
        source_name: Source identifier
        aquifer_type: Aquifer type
        method: Method name

    Returns:
        Total number of chunks created
    """
    if not pdf_dir.exists():
        logger.error(f"Directory not found: {pdf_dir}")
        return 0

    # Find all PDF files
    pdf_files = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return 0

    logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

    total_chunks = 0

    for pdf_file in pdf_files:
        chunks = ingest_pdf_file(
            pdf_path=pdf_file,
            config=config,
            vectorstore=vectorstore,
            chunker=chunker,
            source_name=source_name,
            aquifer_type=aquifer_type,
            method=method
        )
        total_chunks += chunks

    return total_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Custom PDF Documents into HydroAssist",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest all PDFs from a directory
  python scripts/ingest_custom_pdfs.py --pdf-dir ./my_pdfs

  # Ingest a single PDF
  python scripts/ingest_custom_pdfs.py --pdf-file ./document.pdf

  # Specify custom metadata
  python scripts/ingest_custom_pdfs.py --pdf-dir ./my_pdfs \\
      --source "Research Papers" \\
      --aquifer confined \\
      --method "Theis_1935"

  # Use different config
  python scripts/ingest_custom_pdfs.py --pdf-dir ./my_pdfs --config production
        """
    )

    # Required: either pdf-dir or pdf-file
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pdf-dir",
        type=str,
        help="Directory containing PDF files"
    )
    group.add_argument(
        "--pdf-file",
        type=str,
        help="Single PDF file to ingest"
    )

    # Optional metadata
    parser.add_argument(
        "--source",
        type=str,
        default="Custom",
        help="Source name for these documents (default: Custom)"
    )
    parser.add_argument(
        "--aquifer",
        type=str,
        choices=["confined", "unconfined", "leaky", "fractured", "general"],
        default="general",
        help="Aquifer type (default: general)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="multiple",
        help="Method name (e.g., Theis_1935, default: multiple)"
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Configuration name (default, development, production)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Override default chunk size (default: 1000)"
    )

    args = parser.parse_args()

    # Print header
    print("="*60)
    print("HydroAssist Custom PDF Ingestion")
    print("="*60)
    print()

    # Load config
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration: {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Override chunk size if specified
    if args.chunk_size != 1000:
        config.retrieval.chunk_size = args.chunk_size
        logger.info(f"Using custom chunk size: {args.chunk_size}")

    # Initialize components
    try:
        embeddings_manager.get_embeddings(config.retrieval.embedding_model)

        vectorstore = VectorStoreManager(
            persist_directory=config.vectorstore.persist_directory,
            collection_name=config.vectorstore.collection_name
        )

        chunker = MetadataPreservingChunker(
            chunk_size=config.retrieval.chunk_size,
            chunk_overlap=config.retrieval.chunk_overlap
        )
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        sys.exit(1)

    # Display settings
    print(f"Source: {args.source}")
    print(f"Aquifer: {args.aquifer}")
    print(f"Method: {args.method}")
    print(f"Chunk size: {config.retrieval.chunk_size}")
    print()

    # Ingest PDFs
    try:
        if args.pdf_file:
            pdf_path = Path(args.pdf_file)
            if not pdf_path.exists():
                logger.error(f"PDF file not found: {pdf_path}")
                sys.exit(1)

            total_chunks = ingest_pdf_file(
                pdf_path=pdf_path,
                config=config,
                vectorstore=vectorstore,
                chunker=chunker,
                source_name=args.source,
                aquifer_type=args.aquifer,
                method=args.method
            )
        else:
            pdf_dir = Path(args.pdf_dir)
            total_chunks = ingest_pdf_directory(
                pdf_dir=pdf_dir,
                config=config,
                vectorstore=vectorstore,
                chunker=chunker,
                source_name=args.source,
                aquifer_type=args.aquifer,
                method=args.method
            )

        print()
        print("="*60)
        print("Ingestion Complete!")
        print("="*60)
        print(f"Total chunks added: {total_chunks}")
        print()

        # Get collection stats
        stats = vectorstore.get_collection_stats()
        print("Collection Statistics:")
        print(f"  Total chunks in DB: {stats.get('total_chunks', 0)}")

        if 'sources' in stats:
            print("  Sources:")
            for source, count in stats['sources'].items():
                print(f"    - {source}: {count} chunks")

        print()
        logger.info("✅ Custom PDF ingestion completed successfully")

    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()