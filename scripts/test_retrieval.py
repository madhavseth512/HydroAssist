#!/usr/bin/env python3
"""
Quick test of RAG retrieval quality.

Usage:
    python scripts/test_retrieval.py "What is the Theis method?"
    python scripts/test_retrieval.py "Cooper-Jacob assumptions" --top-k 10
    python scripts/test_retrieval.py "confined aquifer" --aquifer confined
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.retriever import retrieve
from src.core.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Test RAG Retrieval Quality"
    )

    parser.add_argument(
        "query",
        help="Search query"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    parser.add_argument(
        "--aquifer",
        help="Filter by aquifer type (confined, unconfined, leaky, fractured)"
    )
    parser.add_argument(
        "--method",
        help="Filter by method name"
    )
    parser.add_argument(
        "--config",
        default="default",
        help="Configuration name (default, development, production)"
    )

    args = parser.parse_args()

    # Build filters
    filters = {}
    if args.aquifer:
        filters["aquifer"] = args.aquifer
    if args.method:
        filters["method"] = args.method

    # Print header
    print(f"\n{'='*80}")
    print(f"RAG Retrieval Test")
    print(f"{'='*80}")
    print(f"Query: {args.query}")
    print(f"Top-K: {args.top_k}")
    print(f"Filters: {filters if filters else 'None'}")
    print(f"{'='*80}\n")

    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        sys.exit(1)

    # Run retrieval
    start = datetime.now()

    try:
        results = retrieve(
            query=args.query,
            config=config,
            top_k=args.top_k,
            metadata_filters=filters if filters else None
        )

        elapsed = (datetime.now() - start).total_seconds()

        if not results:
            print("⚠️  No documents retrieved")
            print("\nPossible reasons:")
            print("  1. Vector database is empty (run: python scripts/ingest.py --all)")
            print("  2. Query doesn't match any indexed content")
            print("  3. Filters are too restrictive")
            sys.exit(0)

        print(f"✅ Retrieved {len(results)} documents in {elapsed:.2f}s")
        print(f"Average score: {sum(r['score'] for r in results) / len(results):.3f}\n")

        # Display results
        for i, doc in enumerate(results, 1):
            metadata = doc['metadata']
            score = doc.get('score', 0)

            print(f"{'─'*80}")
            print(f"[{i}] Score: {score:.3f}")
            print(f"{'─'*80}")

            # Source and citation
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page')

            print(f"Source: {source}", end="")
            if page:
                print(f", p.{page}")
            else:
                print()

            # Metadata
            print(f"Method: {metadata.get('method', 'N/A')}")
            print(f"Aquifer: {metadata.get('aquifer', 'N/A')}")
            print(f"Type: {metadata.get('type', 'N/A')}")
            print(f"Section: {metadata.get('section', 'N/A')}")

            # Content preview
            content = doc['content']
            preview_length = 400
            if len(content) > preview_length:
                print(f"\nContent:\n{content[:preview_length]}...")
            else:
                print(f"\nContent:\n{content}")

            print()

        print(f"{'='*80}\n")

    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
