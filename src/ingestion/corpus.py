"""Turning the papers in `knowledge source/` into chunks, and saying what each one gave.

The script that built the knowledge base did this inline, and silently: a page
under 100 characters was skipped, so a paper that was a pure scan produced
nothing and said nothing. Two papers were absent from the index for months
without a single warning — one of them the recovery paper the calculators cite.

So every document is now measured, and a document that yields far less than its
page count suggests is reported. `verdict` is what the report prints:

    EMPTY   nothing usable — a scan that OCR could not read either
    THIN    a text layer too sparse to be the paper's real content
    OK      a normal text layer

The thresholds come from measuring all 49 papers: a healthy one gives 600-2,400
characters a page, while the five thin ones give 303-370.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

from src.ingestion.processors.pdf_processor import MIN_PAGE_CHARS
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "configs" / "corpus_methods.yaml"

# Every chunk carries `method` as a comma-separated string for display, and one
# flag per method for filtering: ChromaDB metadata holds scalars, not lists, and
# its `where` clause matches values exactly, so "hantush_jacob_1955,hantush_1960"
# would never match a search for "hantush_jacob_1955". A flag does.
METHOD_FLAG_PREFIX = "method_"

UNTAGGED = "multiple"

# The folder a paper sits in is what the corpus knows about its aquifer type.
AQUIFER_BY_FOLDER = {
    "Confined Aquifer": "confined",
    "Unconfined Aquifer": "unconfined",
    "Leaky Confined Aquifer": "leaky",
    "Fractured Aquifers": "fractured",
    "General": "general",
    "Step drawdown test": "general",
}

# Below this, a text layer is captions and page furniture rather than the paper.
THIN_CHARS_PER_PAGE = 500

EMPTY, THIN, OK = "EMPTY", "THIN", "OK"


@dataclass(frozen=True)
class DocumentYield:
    """What one paper contributed, and whether that is believable."""
    name: str
    pages: int
    pages_with_text: int
    characters: int
    ocr_pages: int = 0

    @property
    def chars_per_page(self) -> float:
        return self.characters / self.pages if self.pages else 0.0

    @property
    def verdict(self) -> str:
        if not self.pages_with_text:
            return EMPTY
        if self.chars_per_page < THIN_CHARS_PER_PAGE:
            return THIN
        return OK

    @property
    def needs_attention(self) -> bool:
        return self.verdict in (EMPTY, THIN)

    def describe(self) -> str:
        detail = (f"{self.pages_with_text}/{self.pages} pages, "
                  f"{self.chars_per_page:.0f} chars/page")
        if self.ocr_pages:
            detail += f", {self.ocr_pages} by OCR"
        return f"{self.verdict:<5} {self.name} — {detail}"


def measure(name: str, pages: Sequence[Dict], min_chars: int = MIN_PAGE_CHARS) -> DocumentYield:
    """What a document's extracted pages amount to."""
    usable = [p for p in pages if len(p.get("text") or "") >= min_chars]
    return DocumentYield(
        name=name,
        pages=len(pages),
        pages_with_text=len(usable),
        characters=sum(len(p.get("text") or "") for p in usable),
        ocr_pages=sum(1 for p in pages if p.get("ocr")),
    )


def aquifer_for(pdf_path: Path) -> str:
    """The aquifer type a paper's folder implies."""
    return AQUIFER_BY_FOLDER.get(pdf_path.parent.name, "general")


@dataclass(frozen=True)
class Manifest:
    """Which method each paper is about, from `configs/corpus_methods.yaml`."""
    papers: Dict[str, dict] = field(default_factory=dict)
    implemented: tuple = ()
    other: tuple = ()

    @property
    def vocabulary(self) -> tuple:
        return tuple(self.implemented) + tuple(self.other)

    def methods_for(self, pdf_path: Path) -> List[str]:
        """The method ids a paper covers, or [] when it is not in the manifest —
        which leaves it tagged as it always was rather than guessing."""
        entry = self.papers.get(pdf_path.name)
        return list(entry.get("methods", [])) if entry else []

    def aquifer_for(self, pdf_path: Path) -> str:
        """The manifest may override the folder, which is how the one paper
        filed in two folders gets a stable tag instead of one decided by the
        order the directory happened to be read in."""
        entry = self.papers.get(pdf_path.name) or {}
        return entry.get("aquifer") or aquifer_for(pdf_path)

    def unknown_ids(self) -> Dict[str, List[str]]:
        """Papers tagged with an id the vocabulary does not declare."""
        known = set(self.vocabulary)
        return {name: sorted(set(entry.get("methods", [])) - known)
                for name, entry in self.papers.items()
                if set(entry.get("methods", [])) - known}

    def unreviewed(self) -> List[str]:
        return sorted(name for name, entry in self.papers.items()
                      if not entry.get("reviewed", False))


def load_manifest(path: Path = MANIFEST_PATH) -> Manifest:
    """Read the manifest. A missing file is not fatal: ingestion then tags every
    paper 'multiple', which is what it did before the manifest existed."""
    if not path.exists():
        logger.warning(f"No method manifest at {path}; papers will be tagged "
                       f"'{UNTAGGED}'.")
        return Manifest()

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    vocabulary = data.get("vocabulary", {}) or {}
    return Manifest(
        papers=data.get("papers", {}) or {},
        implemented=tuple(vocabulary.get("implemented", []) or []),
        other=tuple(vocabulary.get("other", []) or []),
    )


# ── What kind of text a chunk is ──────────────────────────────────────────────
#
# A paper is not all argument. Its reference list, its contents page and its
# ordering boilerplate were chunked and embedded exactly like its derivations,
# and they compete on equal terms: a question about the Theis assumptions once
# came back citing Walton's bibliography.
#
# The separation is sharper than it looks. A bibliography writes "Walton, W.C.,
# 1970, Groundwater resource evaluation"; an argument writes "Neuman (1974)
# showed". Only the first is matched here, which is why body text discussing
# many papers does not trip it — checked against all 4,017 chunks before the
# thresholds were fixed.

BODY, REFERENCES, CONTENTS = "body", "references", "contents"

# "Surname, A.B., 1974" and "Surname, A.B. and Other, C., 1974" — a bibliography
# entry, not a citation inside a sentence.
_BIBLIOGRAPHY_ENTRY = re.compile(
    r"[A-Z][a-z]+,\s+[A-Z]\.(?:[A-Z]\.)?,?\s*(?:and\s+[^,]{2,30},\s*)?(?:19|20)\d\d")

# "Introduction ................ 7" — a contents page.
_DOTTED_LEADER = re.compile(r"\.{5,}\s*\d+")

# Entries per 1,000 characters. At 1.5 the lowest-scoring chunks in the corpus
# are still reference lists; body text sits far below, because it cites in
# prose. Two entries minimum, so a short chunk quoting one paper is safe.
_BIBLIOGRAPHY_DENSITY = 1.5
_BIBLIOGRAPHY_MINIMUM = 2
_CONTENTS_MINIMUM = 3


def classify_matter(text: str) -> str:
    """Whether a chunk is the paper's argument or its apparatus.

    Conservative by design: anything unrecognised is BODY. Wrongly dropping a
    derivation costs an answer, while wrongly keeping a reference list costs a
    line in a source list.
    """
    if not text:
        return BODY

    if len(_DOTTED_LEADER.findall(text)) >= _CONTENTS_MINIMUM:
        return CONTENTS

    entries = len(_BIBLIOGRAPHY_ENTRY.findall(text))
    density = entries / max(len(text), 1) * 1000
    if entries >= _BIBLIOGRAPHY_MINIMUM and density >= _BIBLIOGRAPHY_DENSITY:
        return REFERENCES

    return BODY


def method_flags(methods: Sequence[str]) -> Dict[str, bool]:
    """One metadata flag per method, which is what a `where` clause can match."""
    return {f"{METHOD_FLAG_PREFIX}{method}": True for method in methods}


def build_documents(
    pdf_path: Path,
    pages: Sequence[Dict],
    methods: Optional[Sequence[str]] = None,
    aquifer: Optional[str] = None,
    min_chars: int = MIN_PAGE_CHARS,
) -> List[Dict]:
    """The per-page documents for one paper, ready to chunk.

    `methods` is the method ids the paper covers. A paper the manifest does not
    name keeps the 'multiple' tag the corpus has always carried, so an untagged
    paper is merely unfiltered rather than wrongly filtered.
    """
    aquifer = aquifer or aquifer_for(pdf_path)
    folder = pdf_path.parent.name
    methods = list(methods or [])
    method_tag = ",".join(methods) if methods else UNTAGGED

    documents = []
    for page in pages:
        text = page.get("text") or ""
        if len(text) < min_chars:
            continue
        page_number = page["page"]
        doc_id = f"{pdf_path.stem}_page_{page_number}"
        documents.append({
            "text": text,
            "doc_id": doc_id,
            "metadata": {
                "source": f"Custom_{folder}",
                "type": "research_paper",
                "aquifer": aquifer,
                "method": method_tag,
                "page": page_number,
                "section": f"{pdf_path.stem} - Page {page_number}",
                "doc_id": doc_id,
                "has_equations": page.get("has_equations", False),
                "confidence": 1.0,
                "ocr": bool(page.get("ocr")),
                "matter": classify_matter(text),
                **method_flags(methods),
            },
        })
    return documents


def report(yields: Sequence[DocumentYield]) -> str:
    """The end-of-run summary. Documents needing attention come first, because
    that is the only part anyone reads."""
    if not yields:
        return "No documents processed."

    flagged = sorted((y for y in yields if y.needs_attention),
                     key=lambda y: y.chars_per_page)
    lines = [f"{len(yields)} documents, "
             f"{sum(y.characters for y in yields):,} characters, "
             f"{sum(y.ocr_pages for y in yields)} pages read by OCR"]

    if flagged:
        lines.append(f"\n{len(flagged)} need attention:")
        lines.extend("  " + y.describe() for y in flagged)
    else:
        lines.append("Every document yielded a usable text layer.")

    return "\n".join(lines)
