"""Narrow a search to the papers about the method being asked about.

Every chunk used to be tagged "multiple", so a question about Hantush searched
the whole corpus and competed with 483 chunks of Neuman. Now each chunk carries
a flag per method, and this module turns a question into the filter that matches
them.

Two things keep it from doing harm:

- **It only fires when a question names a method.** General questions — "what is
  transmissivity?" — search everything, as before.
- **The retriever falls back.** If a filtered search finds nothing, it retries
  unfiltered, so a wrong guess costs a little time rather than an answer.

The vocabulary is the manifest's, not the calculator's, because the corpus
covers many methods HydroAssist cannot run and those are exactly the questions
where the literature has to answer on its own.
"""

import re
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

from src.ingestion.corpus import METHOD_FLAG_PREFIX, load_manifest
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Ids that name no author — filtering on them would be filtering on nothing.
_NOT_AUTHORS = {"general_reference", "horizontal_well", "observation_well_response",
                "sinusoidal_pumping", "aqtesolv_guide", "wtaq_1999"}


@lru_cache(maxsize=1)
def _vocabulary() -> tuple:
    return load_manifest().vocabulary


def _authors(method_id: str) -> List[str]:
    """The surnames in an id: theis_recovery_1935 -> ['theis'],
    neuman_witherspoon_1969 -> ['neuman', 'witherspoon']."""
    parts = [p for p in method_id.split("_") if not p.isdigit()]
    return [p for p in parts if len(p) > 3 and p not in ("recovery", "jacob", "root")]


def _year(method_id: str) -> Optional[str]:
    match = re.search(r"(1[89]\d{2}|20\d{2})$", method_id)
    return match.group(1) if match else None


def methods_in(question: str, vocabulary: Optional[Sequence[str]] = None) -> List[str]:
    """The method ids a question names, most specific first.

    A surname alone matches every paper by that author, which is the useful
    behaviour for "what did Neuman say about delayed yield?". Naming the year as
    well narrows it to that paper.
    """
    if not question:
        return []

    vocabulary = vocabulary if vocabulary is not None else _vocabulary()
    lowered = question.lower()

    matched = []
    for method_id in vocabulary:
        if method_id in _NOT_AUTHORS:
            continue
        authors = _authors(method_id)
        if not authors or not all(author in lowered for author in authors):
            continue
        year = _year(method_id)
        matched.append((1 if year and year in lowered else 0, method_id))

    if not matched:
        return []

    # If the question gave a year, keep only the papers of that year.
    if any(score for score, _ in matched):
        return [method_id for score, method_id in matched if score]

    return [method_id for _, method_id in matched]


def filter_for(method_ids: Sequence[str]) -> Optional[Dict]:
    """A ChromaDB `where` clause matching chunks tagged with any of them."""
    flags = [{f"{METHOD_FLAG_PREFIX}{method_id}": True} for method_id in method_ids]

    if not flags:
        return None
    if len(flags) == 1:
        return flags[0]
    return {"$or": flags}


def filter_for_question(question: str) -> Optional[Dict]:
    """The filter for a question, or None when it names no method."""
    method_ids = methods_in(question)
    if not method_ids:
        return None

    logger.info(f"Question names {', '.join(method_ids)}; filtering to those papers")
    return filter_for(method_ids)
