#!/usr/bin/env python3
"""Build the help index from `docs/help/`.

    python scripts/ingest_help.py            # show what would be indexed
    python scripts/ingest_help.py --write

The consultant knows hydrogeology and nothing about HydroAssist. Asked "where do
I enter the pumping rate?", it searched Theis (1935) for an answer that is not
there. This indexes the application's own documentation so the guide can answer
from it.

A **separate collection**, deliberately. Mixed into the corpus, a help paragraph
would compete with Hantush on a question about leakage, and the consultant's
"not enough information" message — which names the hydrogeology library —
would start describing a library it no longer matches.

Needs `pip install -r requirements-ingest.txt`.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.core.config import load_config
from src.ingestion.processors.chunker import MetadataPreservingChunker
from src.retrieval.vectorstore import VectorStoreManager

HELP_DIR = Path("docs/help")
COLLECTION = "hydroassist_help"

# Help is short and self-contained; a section is the right unit, and a large
# chunk would answer "where is Q" with a page about the whole of Step 2.
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


def sections(text: str, source: str):
    """Split on the markdown headings, which are already the answer-sized unit.

    The document title travels with every section, so a chunk about "The fields"
    still says it belongs to setting up the test.
    """
    title, current, body = source, None, []
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            continue
        if line.startswith("## "):
            if current and body:
                yield title, current, "\n".join(body).strip()
            current, body = line[3:].strip(), []
            continue
        body.append(line)

    if body:
        yield title, current or title, "\n".join(body).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    if not HELP_DIR.exists():
        print(f"Not found: {HELP_DIR.resolve()}")
        return 1

    config = load_config()
    chunker = MetadataPreservingChunker(chunk_size=CHUNK_SIZE,
                                        chunk_overlap=CHUNK_OVERLAP)

    documents = []
    for path in sorted(HELP_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for title, heading, body in sections(text, path.stem):
            if not body:
                continue
            doc_id = f"help_{path.stem}_{heading.lower().replace(' ', '_')[:40]}"
            documents.append({
                "text": f"{title} — {heading}\n\n{body}",
                "doc_id": doc_id,
                "metadata": {
                    "source": "HydroAssist help",
                    "type": "help",
                    "document": title,
                    "section": heading,
                    "doc_id": doc_id,
                    "file": path.name,
                    # Kept so a help chunk can never be mistaken for a paper if
                    # the two are ever read from one place.
                    "matter": "help",
                },
            })

    chunks = chunker.chunk_documents(documents)
    print(f"{len(list(HELP_DIR.glob('*.md')))} files, {len(documents)} sections, "
          f"{len(chunks)} chunks")
    for document in documents:
        print(f"  {document['metadata']['document'][:34]:36s} "
              f"{document['metadata']['section'][:38]}")

    if not args.write:
        print("\nNothing written. Pass --write to build the index.")
        return 0

    store = VectorStoreManager(
        persist_directory=config.vectorstore.persist_directory,
        collection_name=COLLECTION)

    # Rebuilt wholesale: help is small, and stale sections that no longer match
    # the interface are worse than none.
    existing = store.collection.get(include=[])["ids"]
    if existing:
        store.collection.delete(ids=existing)
        print(f"\nRemoved {len(existing)} chunks from the previous build")

    store.add_documents(
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        ids=[chunk.metadata["chunk_id"] for chunk in chunks])

    print(f"Wrote {len(chunks)} chunks to the {COLLECTION!r} collection.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
