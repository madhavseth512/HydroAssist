#!/usr/bin/env python3
"""
Batch ingestion script for processing many PDFs efficiently.

This script processes PDFs in batches to avoid memory issues and
provides progress tracking for large collections.

Usage:
    python scripts/ingest_batch.py --pdf-dir ./data/raw/custom_pdfs --batch-size 10
    python scripts/ingest_batch.py --pdf-dir ./my_pdfs --batch-size 5 --start-from 10
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.processors.pdf_processor import PDFProcessor
from src.ingestion.processors.chunker import MetadataPreservingChunker
from src.retrieval.vectorstore import VectorStoreManager
from src.retrieval.embeddings import embeddings_manager
from src.core.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("BatchIngest")


def process_pdfs_in_batches(
    pdf_files: List[Path],
    config,
    batch_size: int = 10,
    start_from: int = 0,
    source_name: str = "Custom",
    aquifer_type: str = "general",
    method: str = "multiple"
):
    """Process PDFs in batches with progress tracking."""

    # Initialize components once
    logger.info("Initializing components...")
    embeddings_manager.get_embeddings(config.retrieval.embedding_model)

    vectorstore = VectorStoreManager(
        persist_directory=config.vectorstore.persist_directory,
        collection_name=config.vectorstore.collection_name
    )

    chunker = MetadataPreservingChunker(
        chunk_size=config.retrieval.chunk_size,
        chunk_overlap=config.retrieval.chunk_overlap
    )

    pdf_processor = PDFProcessor()

    # Filter PDFs based on start_from
    pdf_files = pdf_files[start_from:]
    total_pdfs = len(pdf_files)

    print(f"\n{'='*70}")
    print(f"Processing {total_pdfs} PDFs in batches of {batch_size}")
    print(f"{'='*70}\n")

    total_chunks = 0
    failed_pdfs = []
    processed_count = 0

    # Process in batches
    for batch_num, i in enumerate(range(0, total_pdfs, batch_size), 1):
        batch = pdf_files[i:i + batch_size]
        batch_start_time = time.time()

        print(f"\n{'─'*70}")
        print(f"BATCH {batch_num}/{(total_pdfs + batch_size - 1) // batch_size}")
        print(f"Processing PDFs {i+1+start_from} to {min(i+len(batch)+start_from, total_pdfs+start_from)}")
        print(f"{'─'*70}\n")

        batch_chunks = 0

        for pdf_idx, pdf_path in enumerate(batch, 1):
            try:
                pdf_start = time.time()
                print(f"[{i+pdf_idx+start_from}/{total_pdfs+start_from}] Processing: {pdf_path.name[:50]}...")

                # Extract text
                pages = pdf_processor.extract_text_with_pages(str(pdf_path))

                if not pages:
                    logger.warning(f"No pages extracted from {pdf_path.name}")
                    continue

                # Prepare documents
                documents = []
                doc_id_base = pdf_path.stem.replace(" ", "_").replace("-", "_")

                for page_data in pages:
                    page_num = page_data['page']
                    text = page_data['text']

                    if not text or len(text) < 100:
                        continue

                    metadata = {
                        'source': source_name,
                        'type': 'theory',
                        'aquifer': aquifer_type,
                        'method': method,
                        'page': page_num,
                        'section': f'Page {page_num}',
                        'doc_id': f'{doc_id_base}_page_{page_num}',
                        'has_equations': page_data['has_equations'],
                        'confidence': 1.0,
                        'filename': pdf_path.name
                    }

                    documents.append({
                        'text': text,
                        'metadata': metadata,
                        'doc_id': f'{doc_id_base}_page_{page_num}'
                    })

                if not documents:
                    logger.warning(f"No valid content in {pdf_path.name}")
                    continue

                # Chunk
                chunks = chunker.chunk_documents(documents)

                # Add to vector store
                vectorstore.add_documents(
                    documents=[c.page_content for c in chunks],
                    metadatas=[c.metadata for c in chunks],
                    ids=[c.metadata['chunk_id'] for c in chunks]
                )

                batch_chunks += len(chunks)
                total_chunks += len(chunks)
                processed_count += 1

                elapsed = time.time() - pdf_start
                print(f"  ✓ {len(pages)} pages → {len(chunks)} chunks ({elapsed:.1f}s)")

            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                failed_pdfs.append((pdf_path.name, str(e)))
                print(f"  ✗ FAILED: {e}")

        # Batch summary
        batch_elapsed = time.time() - batch_start_time
        print(f"\nBatch {batch_num} complete:")
        print(f"  • Processed: {len(batch)} PDFs")
        print(f"  • Chunks added: {batch_chunks}")
        print(f"  • Time: {batch_elapsed/60:.1f} minutes")
        print(f"  • Avg: {batch_elapsed/len(batch):.1f}s per PDF")

    return {
        'total_pdfs': total_pdfs,
        'processed': processed_count,
        'total_chunks': total_chunks,
        'failed': failed_pdfs
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch PDF Ingestion with Progress Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--pdf-dir",
        type=str,
        required=True,
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of PDFs to process per batch (default: 10)"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=0,
        help="Skip first N PDFs (useful for resuming, default: 0)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="Custom",
        help="Source name (default: Custom)"
    )
    parser.add_argument(
        "--aquifer",
        choices=["confined", "unconfined", "leaky", "fractured", "general"],
        default="general",
        help="Aquifer type (default: general)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="multiple",
        help="Method name (default: multiple)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Configuration (default: default)"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Find PDFs
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        logger.error(f"Directory not found: {pdf_dir}")
        sys.exit(1)

    pdf_files = sorted(list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF")))

    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_dir}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"HydroAssist Batch PDF Ingestion")
    print(f"{'='*70}")
    print(f"Directory: {pdf_dir}")
    print(f"Total PDFs found: {len(pdf_files)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Starting from: {args.start_from}")
    print(f"Source: {args.source}")
    print(f"Aquifer: {args.aquifer}")
    print(f"Method: {args.method}")

    # Start processing
    start_time = time.time()

    try:
        results = process_pdfs_in_batches(
            pdf_files=pdf_files,
            config=config,
            batch_size=args.batch_size,
            start_from=args.start_from,
            source_name=args.source,
            aquifer_type=args.aquifer,
            method=args.method
        )

        # Final summary
        total_time = time.time() - start_time

        print(f"\n{'='*70}")
        print(f"INGESTION COMPLETE!")
        print(f"{'='*70}")
        print(f"✓ Successfully processed: {results['processed']}/{results['total_pdfs']} PDFs")
        print(f"✓ Total chunks added: {results['total_chunks']:,}")
        print(f"✓ Total time: {total_time/60:.1f} minutes")
        print(f"✓ Average: {total_time/results['processed']:.1f}s per PDF")

        if results['failed']:
            print(f"\n⚠ Failed PDFs ({len(results['failed'])}):")
            for filename, error in results['failed']:
                print(f"  • {filename}: {error}")

        # Collection stats
        print(f"\n{'─'*70}")
        vectorstore = VectorStoreManager(
            persist_directory=config.vectorstore.persist_directory,
            collection_name=config.vectorstore.collection_name
        )
        stats = vectorstore.get_collection_stats()

        print("Collection Statistics:")
        print(f"  Total chunks in DB: {stats.get('total_chunks', 0):,}")

        if 'sources' in stats:
            print(f"  Sources:")
            for source, count in stats['sources'].items():
                print(f"    - {source}: {count:,} chunks")

        print(f"\n{'='*70}\n")

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        print(f"Progress saved. Resume with: --start-from {args.start_from + processed_count}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Batch ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()