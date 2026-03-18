#!/usr/bin/env python3
"""
CLI tool for ingesting data into HydroAssist knowledge base.

Usage:
    python scripts/ingest.py --source usgs
    python scripts/ingest.py --source aqtesolv
    python scripts/ingest.py --all
    python scripts/ingest.py --all --force-download --validate
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import IngestionPipeline
from src.core.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("IngestCLI")


def main():
    parser = argparse.ArgumentParser(
        description="HydroAssist Data Ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest.py --source usgs
  python scripts/ingest.py --source aqtesolv
  python scripts/ingest.py --all --force-download
        """
    )

    parser.add_argument(
        "--source",
        choices=["usgs", "aqtesolv", "all"],
        default="all",
        help="Data source to ingest (default: all)"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download even if cached"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after ingestion"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Override default chunk size (default: 1000)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Configuration name (default, development, production)"
    )

    args = parser.parse_args()

    # Print header
    print("="*60)
    print("HydroAssist Data Ingestion Pipeline")
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

    # Create pipeline
    try:
        pipeline = IngestionPipeline(config)
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)

    # Run ingestion
    logger.info(f"Starting ingestion: source={args.source}")
    print(f"Source: {args.source}")
    print(f"Force download: {args.force_download}")
    print(f"Validation: {args.validate or config.ingestion.validate_metadata}")
    print()

    try:
        stats = pipeline.ingest(
            source=args.source,
            force_download=args.force_download,
            validate=args.validate or config.ingestion.validate_metadata
        )

        print()
        print("="*60)
        print("Ingestion Complete!")
        print("="*60)
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Sources: {', '.join(stats['sources'])}")
        print(f"Errors: {stats.get('errors', 0)}")
        print()

        # Get collection stats
        collection_stats = pipeline.get_stats()
        print("Collection Statistics:")
        print(f"  Total chunks in DB: {collection_stats.get('total_chunks', 0)}")

        if 'sources' in collection_stats:
            print("  Sources:")
            for source, count in collection_stats['sources'].items():
                print(f"    - {source}: {count} chunks")

        if 'aquifer_types' in collection_stats:
            print("  Aquifer types:")
            for aquifer, count in collection_stats['aquifer_types'].items():
                print(f"    - {aquifer}: {count} chunks")

        print()
        logger.info("✅ Ingestion completed successfully")

    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
