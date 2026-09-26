"""What a retrieved chunk should be called.

Every citation in the app named a folder. The chunk metadata's `source` field
holds one of six values — `Custom_Confined Aquifer`, `Custom_General` and four
more — because that is what the ingestion walk had to hand, and both the
"Sources consulted" list and the citations the model writes inline were built
from it. A reader was told the answer came from "Custom_Fractured Aquifers,
p.5", which identifies nothing they could go and read.

The document's real identity was in the index all along, in `section` and
`doc_id`, as the filename it was chunked from. The trouble with a filename is
that a publisher's is not a name: `1-s2.0-002216948890025X-main.pdf` and
`spe-426-pa.pdf` say nothing either. So three things are tried, in order:

1. **A `title` in the manifest**, `configs/corpus_methods.yaml`, for the papers
   whose filename says nothing.
2. **A cleaned filename**, where it is descriptive. Most are — "Water Resources
   Research - April 1974 - Neuman - Effect of partial penetration…" — and need
   only the publisher's prefixes and suffixes taken off.
3. **The manifest's `note`.** Every paper has a reviewed entry and half carry a
   note, but those were written as annotations rather than names ("scanned;
   recovered by OCR"), so they are a fallback and not the first choice.
4. **The filename as it stands**, which is at least true.

Links are only offered where one can be derived rather than guessed. A
ScienceDirect filename encodes the article's PII, which maps to a DOI by a fixed
rule, and the manifest can name a URL for anything else. Most of the corpus is
paywalled, so a citation says so plainly instead of implying a link that would
not open.

Nothing here rebuilds the index: resolution happens at query time, on the
metadata that is already stored, so the 34 MB vector store is untouched.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

MANIFEST = Path("configs/corpus_methods.yaml")

# Publisher furniture: a download's prefix, and the "-main" every ScienceDirect
# PDF carries.
_PREFIXES = (
    re.compile(r"^1-s2\.0-"),
    re.compile(r"^52485_"),
    re.compile(r"^PDF_Ppr_"),
)
_SUFFIXES = (re.compile(r"-main$"), re.compile(r"\s*\(\d+\)$"))

# 1-s2.0-002216948890025X-main.pdf holds the article's PII, which becomes a DOI
# by a fixed rule: 10.1016/ then SSSS-SSSS(YY)NNNNN-C.
_SCIENCEDIRECT = re.compile(r"^1-s2\.0-([0-9X]{16})-main$", re.I)

# "<stem>_page_<n>" is how the ingestion pipeline names a chunk's document.
_DOC_ID = re.compile(r"^(?P<stem>.+)_page_(?P<page>\d+)$")


@dataclass(frozen=True)
class Source:
    """A retrieved chunk's provenance, as a reader would want it."""
    document: str
    page: Optional[int] = None
    url: str = ""
    method: str = ""
    aquifer: str = ""
    # True for the app's own help text, which is not a paper and needs no link.
    is_help: bool = False

    @property
    def citation(self) -> str:
        """The short form, as it appears inline in an answer."""
        return f"[{self.document}, p.{self.page}]" if self.page \
            else f"[{self.document}]"


@lru_cache(maxsize=1)
def _manifest() -> Dict[str, dict]:
    """The reviewed entry per paper, or {} if the manifest cannot be read.

    Cached: a chat turn resolves five chunks, and the file does not change while
    the server runs. A missing or broken manifest costs better names, never an
    answer, so every failure here is silent by design.
    """
    try:
        import yaml

        with MANIFEST.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        papers = loaded.get("papers") or {}
        # Keyed by stem, since that is what a chunk's metadata carries.
        return {Path(name).stem: entry or {} for name, entry in papers.items()}
    except Exception:
        return {}


def _stem_and_page(metadata: dict) -> tuple:
    """The document a chunk came from, and which page of it."""
    page = metadata.get("page")
    try:
        page = int(page) if page is not None else None
    except (TypeError, ValueError):
        page = None

    doc_id = str(metadata.get("doc_id") or "")
    match = _DOC_ID.match(doc_id)
    if match:
        return match.group("stem"), page or int(match.group("page"))

    # "<stem> - Page 3" is the other form the index holds.
    section = str(metadata.get("section") or "")
    if " - Page" in section:
        return section.split(" - Page")[0].strip(), page
    return section.strip() or doc_id, page


def _readable(stem: str) -> str:
    """A filename with the publisher's furniture taken off."""
    name = stem
    for pattern in _PREFIXES:
        name = pattern.sub("", name)
    for pattern in _SUFFIXES:
        name = pattern.sub("", name)
    name = name.replace("_", " ").strip()
    # Two spaces occur in several of these filenames and read as a typo.
    return re.sub(r"\s{2,}", " ", name) or stem


# Two real words is the line between a name and a catalogue number: "SPE-9289
# Agarwal buildup" reads as a paper, "WR024i010p01796" and "spe-426-pa" do not,
# and those are the ones the manifest has to name.
_WORD = re.compile(r"[A-Za-z]{3,}")


def _is_descriptive(name: str) -> bool:
    """Whether this filename tells a reader what the document is."""
    return len(_WORD.findall(name)) >= 2


def _derived_url(stem: str) -> str:
    """A DOI link where the filename encodes one, and "" otherwise.

    Deliberately narrow. A link that 404s is worse than no link, so this covers
    only the case where the identifier is in the filename by construction.
    """
    match = _SCIENCEDIRECT.match(stem)
    if not match:
        return ""
    pii = match.group(1).upper()
    doi = f"10.1016/{pii[0:4]}-{pii[4:8]}({pii[8:10]}){pii[10:15]}-{pii[15]}"
    return f"https://doi.org/{doi}"


def resolve(metadata: dict) -> Source:
    """What to call the document a chunk came from."""
    metadata = metadata or {}

    # The app's own help text: a document and a section, not a paper.
    if metadata.get("type") == "help":
        document = str(metadata.get("document") or "HydroAssist help")
        section = str(metadata.get("section") or "")
        return Source(document=f"{document} — {section}".strip(" —"),
                      is_help=True)

    stem, page = _stem_and_page(metadata)
    entry = _manifest().get(stem, {})

    readable = _readable(stem)
    document = (entry.get("title")
                or (readable if _is_descriptive(readable) else "")
                or entry.get("note")
                or readable
                or str(metadata.get("source") or "Unknown"))
    return Source(
        document=document,
        page=page,
        url=str(entry.get("url") or "") or _derived_url(stem),
        method=str(metadata.get("method") or ""),
        aquifer=str(metadata.get("aquifer") or ""),
    )
