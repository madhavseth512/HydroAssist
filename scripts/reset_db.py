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

# This script prints ✅ and ❌, which a cp1252 Windows console cannot encode: the
# UnicodeEncodeError aborted the run and left the database in place while the
# caller went on believing it had been deleted.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.core.config import load_config


def _release_chromadb() -> None:
    """Close ChromaDB's handles on the store, so it can be deleted."""
    try:
        import gc

        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
        gc.collect()
    except Exception:
        # Nothing opened it, or this ChromaDB has no such cache: the delete
        # below will report anything that actually blocks it.
        pass


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

        # No second prompt. --confirm is already required — the script refuses
        # to run without it — so asking again only meant that from a script or a
        # rebuild pipeline input() raised EOFError, the delete never happened,
        # and the next command ingested into a database the caller believed was
        # empty. The flag is the confirmation.

        # Let go of the store before deleting it. Reading the stats above opened
        # it, and on Windows an open file cannot be removed: the delete failed
        # with WinError 32 while reporting the database as reset.
        _release_chromadb()

        print(f"Deleting: {db_path}")
        try:
            shutil.rmtree(db_path)
            print("✅ Database reset complete")
        except Exception as e:
            print(f"❌ Failed to delete database: {e}")
            print("   Close anything else using the store, or delete the "
                  "directory by hand.")
            sys.exit(1)

    else:
        print("ℹ️  No database found (already empty or never created)")

    print()
    print("To rebuild the database, run:")
    print("  python scripts/ingest.py --all")
    print()


if __name__ == "__main__":
    main()
