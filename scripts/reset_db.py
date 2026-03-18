#!/usr/bin/env python3
"""
Reset the vector database (useful for development).

Usage:
    python scripts/reset_db.py --confirm
"""

import argparse
import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Reset Vector Database",
        epilog="WARNING: This will permanently delete all ingested data!"
    )

    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm deletion (required for safety)"
    )
    parser.add_argument(
        "--config",
        default="default",
        help="Configuration name (default, development, production)"
    )

    args = parser.parse_args()

    if not args.confirm:
        print("❌ Must use --confirm flag to delete database")
        print()
        print("This operation will PERMANENTLY DELETE all ingested data!")
        print()
        print("Usage: python scripts/reset_db.py --confirm")
        sys.exit(1)

    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        sys.exit(1)

    db_path = Path(config.vectorstore.persist_directory)

    print("="*60)
    print("Vector Database Reset")
    print("="*60)
    print(f"Database path: {db_path}")
    print(f"Collection: {config.vectorstore.collection_name}")
    print()

    if db_path.exists():
        # Show stats before deletion
        try:
            from src.retrieval.vectorstore import VectorStoreManager

            vm = VectorStoreManager(
                persist_directory=str(db_path),
                collection_name=config.vectorstore.collection_name
            )

            stats = vm.get_collection_stats()
            print(f"Current database contains:")
            print(f"  Total chunks: {stats.get('total_chunks', 0)}")

            if 'sources' in stats:
                print(f"  Sources: {', '.join(stats['sources'].keys())}")

            print()
        except Exception as e:
            print(f"Could not read stats: {e}")
            print()

        # Confirm again
        response = input("Are you sure you want to DELETE this database? Type 'yes' to confirm: ")

        if response.lower() != 'yes':
            print("❌ Aborted")
            sys.exit(0)

        # Delete
        print(f"Deleting: {db_path}")
        try:
            shutil.rmtree(db_path)
            print("✅ Database reset complete")
        except Exception as e:
            print(f"❌ Failed to delete database: {e}")
            sys.exit(1)

    else:
        print("ℹ️  No database found (already empty or never created)")

    print()
    print("To rebuild the database, run:")
    print("  python scripts/ingest.py --all")
    print()


if __name__ == "__main__":
    main()
