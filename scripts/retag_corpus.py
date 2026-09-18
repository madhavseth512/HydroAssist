#!/usr/bin/env python3
"""Apply `configs/corpus_methods.yaml` to the chunks already in the store.

    python scripts/retag_corpus.py             # show what would change
    python scripts/retag_corpus.py --write     # change it

Changing a paper's methods does not change a single word of its text, so
re-running the ingest would re-embed 4,017 chunks to write a handful of metadata
keys — minutes of work, and a rebuilt binary to commit, for a label. This
rewrites the labels in place.

It does not add or remove chunks. A paper added to `knowledge source/`, or an
OCR change, still needs `scripts/ingest_knowledge_source.py`.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.core.config import load_config
from src.ingestion import corpus
from src.retrieval.vectorstore import VectorStoreManager

# doc_id is "<pdf stem>_page_<n>", so this recovers the paper a chunk came from.
def _stem_of(doc_id: str) -> str:
    return doc_id.rsplit("_page_", 1)[0] if "_page_" in doc_id else doc_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true",
                        help="write the new tags (otherwise only report)")
    args = parser.parse_args()

    manifest = corpus.load_manifest()
    if not manifest.papers:
        print("No manifest found; nothing to apply.")
        return 1

    unknown = manifest.unknown_ids()
    if unknown:
        for name, ids in unknown.items():
            print(f"Manifest error: {name} uses undeclared id(s) {', '.join(ids)}")
        return 1

    # A paper is known by its filename in the manifest and by its stem in the
    # store, so index the manifest by stem once.
    by_stem = {Path(name).stem: name for name in manifest.papers}

    config = load_config()
    store = VectorStoreManager(
        persist_directory=config.vectorstore.persist_directory,
        collection_name=config.vectorstore.collection_name)
    held = store.collection.get(include=["metadatas"])
    ids, metadatas = held["ids"], held["metadatas"]

    print(f"{len(ids):,} chunks in the store, {len(manifest.papers)} papers in "
          f"the manifest\n")

    changed_ids, changed_metadatas = [], []
    per_paper, unmatched = Counter(), Counter()

    for chunk_id, metadata in zip(ids, metadatas):
        stem = _stem_of(str(metadata.get("doc_id", "")))
        name = by_stem.get(stem)
        if name is None:
            unmatched[stem] += 1
            continue

        methods = list(manifest.papers[name].get("methods", []))
        wanted = {"method": ",".join(methods) if methods else corpus.UNTAGGED,
                  **corpus.method_flags(methods)}

        # Flags this chunk carries that the manifest no longer gives it. Setting
        # a key to None is how ChromaDB is told to drop it.
        stale = {key: None for key in metadata
                 if key.startswith(corpus.METHOD_FLAG_PREFIX) and key not in wanted}

        if all(metadata.get(key) == value for key, value in wanted.items()) and not stale:
            continue

        updated = {**metadata, **wanted, **stale}
        changed_ids.append(chunk_id)
        changed_metadatas.append(updated)
        per_paper[name] += 1

    if unmatched:
        print("Chunks whose paper is not in the manifest (left untouched):")
        for stem, count in unmatched.most_common():
            print(f"  {count:5d}  {stem[:60]}")
        print()

    if not changed_ids:
        print("Every chunk already matches the manifest.")
        return 0

    print(f"{len(changed_ids):,} chunks would change, across {len(per_paper)} paper(s):")
    for name, count in per_paper.most_common():
        print(f"  {count:5d}  {name[:56]} -> "
              f"{', '.join(manifest.papers[name].get('methods', []))[:60]}")

    if not args.write:
        print("\nNothing written. Pass --write to apply.")
        return 0

    # ChromaDB caps how much one update may carry.
    batch = 1000
    for start in range(0, len(changed_ids), batch):
        store.collection.update(ids=changed_ids[start:start + batch],
                                metadatas=changed_metadatas[start:start + batch])

    print(f"\nWrote {len(changed_ids):,} chunks. Verifying...")

    after = store.collection.get(ids=changed_ids[:50], include=["metadatas"])
    for metadata in after["metadatas"]:
        name = by_stem[_stem_of(str(metadata["doc_id"]))]
        expected = ",".join(manifest.papers[name].get("methods", [])) or corpus.UNTAGGED
        if metadata.get("method") != expected:
            print(f"  MISMATCH on {metadata['doc_id']}: "
                  f"{metadata.get('method')!r} != {expected!r}")
            return 1
    print("  the sample reads back as expected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
