#!/usr/bin/env python3
"""Re-record what retrieval returns, after a deliberate change to the corpus.

    python scripts/rebaseline_retrieval.py            # show what changed
    python scripts/rebaseline_retrieval.py --write    # and record it

`tests/fixtures/retrieval_baseline.json` pins the top-5 chunks for a fixed set
of questions. That test exists to catch an *accidental* change — a different
embedder, a half-rebuilt index. Rebuilding the corpus on purpose changes those
answers legitimately, and the baseline then has to be re-recorded.

Printing the difference first is the point: a rebuild that was supposed to add
the two scanned papers should show movement on recovery questions and little
anywhere else. Anything wider is worth understanding before it is recorded.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASELINE = Path("tests/fixtures/retrieval_baseline.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help="record the new results as the baseline")
    parser.add_argument("--reason", default="",
                        help="why the corpus changed, kept in the fixture")
    args = parser.parse_args()

    from src.core.config import load_config
    from src.retrieval.retriever import Retriever
    from src.retrieval.vectorstore import VectorStoreManager

    existing = json.loads(BASELINE.read_text(encoding="utf-8"))
    queries = existing["queries"]

    config = load_config()
    retriever = Retriever(config)
    store = VectorStoreManager(
        persist_directory=config.vectorstore.persist_directory,
        collection_name=config.vectorstore.collection_name)
    total_chunks = store.get_collection_stats().get("total_chunks", 0)

    print(f"{len(queries)} queries against {total_chunks:,} chunks\n")

    captured, moved = {}, 0
    for query, expected in queries.items():
        hits = retriever.retrieve(query, top_k=5, similarity_threshold=0.0)
        captured[query] = [{"chunk_id": h["metadata"].get("chunk_id"),
                            "score": h["score"]} for h in hits]

        before = [e["chunk_id"] for e in expected]
        after = [h["chunk_id"] for h in captured[query]]
        if before == after:
            print(f"  unchanged  {query}")
            continue

        moved += 1
        kept = len(set(before) & set(after))
        print(f"  CHANGED    {query}")
        print(f"             {kept}/5 of the same chunks; top result "
              f"{'the same' if before[0] == after[0] else 'different'}")
        for chunk_id in after:
            if chunk_id not in before:
                print(f"             + {chunk_id}")

    print(f"\n{moved} of {len(queries)} queries changed.")

    if not args.write:
        print("Nothing written. Pass --write to record this as the baseline.")
        return 0

    provenance = (
        f"{existing.get('_provenance', '')}\n"
        f"Re-recorded {date.today().isoformat()} against {total_chunks:,} chunks"
        f"{' — ' + args.reason if args.reason else ''}. The query embedder is "
        f"unchanged; the corpus is not."
    ).strip()

    BASELINE.write_text(
        json.dumps({"_provenance": provenance, "queries": captured}, indent=2) + "\n",
        encoding="utf-8")
    print(f"Recorded in {BASELINE}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
