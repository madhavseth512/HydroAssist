#!/usr/bin/env python3
"""Build the knowledge base from the papers in `knowledge source/`.

    python scripts/ingest_knowledge_source.py --dry-run          # measure only
    python scripts/ingest_knowledge_source.py                    # OCR where needed
    python scripts/ingest_knowledge_source.py --only Agarwal     # one paper
    python scripts/ingest_knowledge_source.py --ocr never        # text layers only

Needs `pip install -r requirements-ingest.txt`, and Tesseract for OCR.

What changed, and why: this script used to skip any page under 100 characters
without a word of complaint, so two papers that are pure scans contributed
nothing to the index and nobody noticed for months. It now recognises the pages
that have no text layer, and prints what every document gave, naming the ones
that gave too little.

The chunking and metadata live in `src/ingestion/corpus.py`, where they are
tested; this file is the command-line wrapper around them.
"""

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Half the papers are journal downloads whose filenames carry typographic
# characters — "Neuman - Effect of partial penetration…" uses a non-breaking
# hyphen. On a Windows console, which is cp1252, printing one raises
# UnicodeEncodeError and takes the whole run down after the work is done.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.core.config import load_config
from src.ingestion import corpus
from src.ingestion.processors.chunker import MetadataPreservingChunker
from src.ingestion.processors.ocr import OCRUnavailable, check_tooling
from src.ingestion.processors.pdf_processor import OCR_AUTO, OCR_MODES, PDFProcessor
from src.retrieval.vectorstore import VectorStoreManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

KNOWLEDGE_SOURCE = Path("./knowledge source")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", metavar="TEXT",
                        help="only papers whose filename contains TEXT")
    parser.add_argument("--ocr", choices=OCR_MODES, default=OCR_AUTO,
                        help="how to read pages with no text layer (default: auto)")
    parser.add_argument("--dry-run", action="store_true",
                        help="measure and report, without writing to the store")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.ocr != "never":
        tooling = check_tooling()
        print(f"OCR: {tooling.instructions}")
        if not tooling.available:
            print("Continuing would leave scanned papers empty. Install the tooling "
                  "above, or pass --ocr never to accept text layers only.")
            return 1

    if not KNOWLEDGE_SOURCE.exists():
        print(f"Not found: {KNOWLEDGE_SOURCE.resolve()}")
        return 1

    manifest = corpus.load_manifest()
    if manifest.papers:
        unknown = manifest.unknown_ids()
        if unknown:
            for name, ids in unknown.items():
                print(f"Manifest error: {name} uses undeclared id(s) {', '.join(ids)}")
            return 1
        print(f"Method manifest: {len(manifest.papers)} papers, "
              f"{len(manifest.unreviewed())} not yet reviewed")

    config = load_config()
    processor = PDFProcessor()
    chunker = MetadataPreservingChunker(
        chunk_size=config.retrieval.chunk_size,
        chunk_overlap=config.retrieval.chunk_overlap,
    )
    store = None if args.dry_run else VectorStoreManager(
        persist_directory=config.vectorstore.persist_directory,
        collection_name=config.vectorstore.collection_name,
    )

    pdfs = sorted(KNOWLEDGE_SOURCE.rglob("*.pdf"))
    if args.only:
        pdfs = [p for p in pdfs if args.only.lower() in p.name.lower()]
    if not pdfs:
        print("No PDFs matched.")
        return 1

    print(f"{len(pdfs)} paper(s){' — dry run, nothing will be written' if args.dry_run else ''}\n")

    yields, chunk_total, seen_digests, failures = [], 0, {}, []

    for index, pdf in enumerate(pdfs, 1):
        print(f"[{index}/{len(pdfs)}] {pdf.name}", flush=True)

        digest = hashlib.md5(pdf.read_bytes()).hexdigest()
        if digest in seen_digests:
            # Identical files share a doc_id, so the second would overwrite the
            # first and its folder would silently decide the aquifer tag.
            print(f"      duplicate of {seen_digests[digest]} — skipped")
            continue
        seen_digests[digest] = pdf.name

        try:
            pages = processor.extract_text_with_pages(str(pdf), ocr=args.ocr)
        except OCRUnavailable as error:
            print(f"      OCR unavailable: {error}")
            return 1
        except Exception as error:                      # a corrupt or locked PDF
            print(f"      FAILED to read: {error}")
            failures.append(pdf.name)
            continue

        measured = corpus.measure(pdf.name, pages)
        yields.append(measured)

        methods = manifest.methods_for(pdf)
        documents = corpus.build_documents(
            pdf, pages, methods=methods, aquifer=manifest.aquifer_for(pdf))
        chunks = chunker.chunk_documents(documents)
        chunk_total += len(chunks)

        tags = ", ".join(methods) if methods else "multiple (untagged)"
        print(f"      {measured.describe()} -> {len(chunks)} chunks [{tags}]")

        if store is not None and chunks:
            store.add_documents(
                documents=[c.page_content for c in chunks],
                metadatas=[c.metadata for c in chunks],
                ids=[c.metadata["chunk_id"] for c in chunks],
            )

    print("\n" + "=" * 70)
    print(corpus.report(yields))
    print(f"{chunk_total:,} chunks {'would be' if args.dry_run else ''} written")
    if failures:
        print(f"{len(failures)} paper(s) could not be read: {', '.join(failures)}")
    print("=" * 70)

    if store is not None:
        stats = store.get_collection_stats()
        print(f"\nThe store now holds {stats['total_chunks']:,} chunks.")

    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
