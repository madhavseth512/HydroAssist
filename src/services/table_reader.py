"""Reading a table of readings, in whatever form it arrives.

The workspace accepted CSV and nothing else, which is not how field data
travels. An old software export is tab-delimited `.txt`; a modern one is an
Excel sheet; and most people already have the numbers open in a spreadsheet and
would rather paste two columns than save a file first.

Only the reading changes. Everything downstream — `CSVInspector.inspect`,
`DataFormatter.format`, `DataValidator.validate` — already takes a DataFrame and
neither knows nor cares where it came from, so this module is the whole of the
new surface.

What it does not do is guess silently. A file that cannot be read says which
formats were tried, and a spreadsheet with several sheets says which one was
used, because a wrong sheet produces a plausible fit of the wrong data.
"""

import io
from typing import List, Optional, Tuple

import pandas as pd

from src.services import logger_import

# Extensions we recognise, and what to say when none matches.
DELIMITED = (".csv", ".txt", ".tsv", ".dat", ".prn")
LOGGER = (".lev", ".mon", ".xle")
EXCEL = (".xlsx", ".xlsm", ".xltx")
SUPPORTED = DELIMITED + EXCEL + LOGGER

# Not read here, and not readable as a table of numbers either: a photograph goes
# through `table_ocr`, which proposes a grid rather than producing readings.
PICTURES = (".png", ".jpg", ".jpeg", ".heic", ".tif", ".tiff", ".webp", ".bmp")

# Pasted text arrives without a filename; this stands in for one.
PASTED = "pasted.txt"


class UnreadableTable(ValueError):
    """The bytes could not be read as a table of readings."""


def _suffix(filename: str) -> str:
    name = (filename or "").strip().lower()
    dot = name.rfind(".")
    return name[dot:] if dot > 0 else ""


def _read_delimited(raw: bytes) -> Tuple[pd.DataFrame, List[str]]:
    """CSV, TSV or whitespace-delimited text, and what was skipped to read it.

    `sep=None` with the Python engine sniffs the delimiter, which is what lets
    one path serve a comma file, a tab file and a pasted column pair. The
    encoding falls back to latin-1: field software still writes it, and a single
    degree sign in a header should not cost someone their upload.
    """
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        body, notes = _past_the_comments(text)
        if not body.strip():
            raise UnreadableTable(
                "the file holds only comments — no header row and no readings")
        try:
            return pd.read_csv(io.StringIO(body), sep=None, engine="python",
                               skip_blank_lines=True), notes
        except Exception as exc:
            raise UnreadableTable(str(exc)) from exc
    raise UnreadableTable("the text is in an encoding this cannot read")


# A leading `#` marks a comment in nearly every ASCII data file a hydrogeologist
# meets: USGS exports carry provenance that way, and so do this project's own
# field-data fixtures. Reading the first such line as the header row is what
# happened before — the prose was split on the commas inside it, the real header
# became a row of data, and the upload failed with pandas' "[''] not in index".
_COMMENT = "#"

# Enough of the provenance to be worth reading, not so much that it buries the
# conversions underneath it.
_COMMENTS_SHOWN = 6


def _past_the_comments(text: str) -> Tuple[str, List[str]]:
    """The file from its header row on, and the comments that preceded it.

    Only the leading block is skipped. A `#` further down is left alone: it is
    more likely part of a well name — "MW#3" — than a comment, and truncating a
    line of readings at it would lose data silently.
    """
    lines = text.splitlines()
    comments, start = [], 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            start = index + 1
            continue
        if stripped.startswith(_COMMENT):
            comments.append(stripped.lstrip("# ").strip())
            start = index + 1
            continue
        start = index
        break

    if not comments:
        return text, []

    notes = [f"Skipped {len(comments)} comment line(s) above the header row"]
    # The comments usually say where the readings came from, which is worth
    # putting in front of the user rather than dropping on the floor.
    notes.extend(f"From the file: {comment}"
                 for comment in comments[:_COMMENTS_SHOWN] if comment)
    if len(comments) > _COMMENTS_SHOWN:
        notes.append(f"…and {len(comments) - _COMMENTS_SHOWN} more comment line(s)")
    return "\n".join(lines[start:]), notes


def _read_excel(raw: bytes, sheet: Optional[str]) -> Tuple[pd.DataFrame, List[str]]:
    """One sheet of a workbook, and a note saying which.

    Named sheet wins. Otherwise the first sheet that looks like readings is
    used, rather than simply the first sheet: a real workbook often opens with a
    cover page or a sheet of site notes, and failing on that would send people
    back to Excel to delete a tab before HydroAssist would speak to them.
    """
    try:
        book = pd.ExcelFile(io.BytesIO(raw))
    except Exception as exc:
        raise UnreadableTable(str(exc)) from exc

    names = list(book.sheet_names)
    if not names:
        raise UnreadableTable("the workbook has no sheets")

    if sheet is not None:
        if sheet not in names:
            raise UnreadableTable(
                f"no sheet named {sheet!r}; this workbook has {', '.join(names)}")
        return book.parse(sheet), _sheet_note(sheet, names)

    for candidate in names:
        frame = _tidy(book.parse(candidate))
        if len(frame.columns) >= 2 and not frame.empty:
            return frame, _sheet_note(candidate, names)

    raise UnreadableTable(
        f"no sheet holds a table of at least two columns "
        f"({', '.join(names)}). Name the sheet to use, or save it as CSV.")


def _sheet_note(chosen: str, names: List[str]) -> List[str]:
    """Which sheet was read, whenever there was a choice.

    Never silent: the wrong sheet fits the wrong data perfectly well, and the
    user would have no way to tell.
    """
    if len(names) <= 1:
        return []
    return [f"Read sheet {chosen!r} of {len(names)} ({', '.join(names)})"]


def read_table(raw: bytes, filename: str = "",
               sheet: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """A DataFrame from uploaded or pasted bytes, plus notes for the user.

    Args:
        raw: the file's bytes, or pasted text encoded as UTF-8.
        filename: used only to choose a reader. Empty means delimited text.
        sheet: which sheet of a workbook; the first by default.

    Raises:
        UnreadableTable: with a message naming what was tried.
    """
    if not raw or not raw.strip():
        raise UnreadableTable("the file is empty")

    suffix = _suffix(filename)

    if suffix in EXCEL:
        frame, notes = _read_excel(raw, sheet)
    elif suffix in LOGGER or (suffix in DELIMITED
                              and logger_import.looks_like_logger_export(raw)):
        # A transducer export: a metadata block, a clock, and a water level
        # rather than elapsed seconds and drawdown. Recognised by its header
        # even when it is named .csv, which is how In-Situ writes it.
        try:
            readings = logger_import.read_logger_export(raw)
        except logger_import.NotALoggerFile as error:
            raise UnreadableTable(str(error)) from error
        frame, notes = readings.df, list(readings.notes)
    elif suffix in DELIMITED or not suffix:
        frame, notes = _read_delimited(raw)
    else:
        raise UnreadableTable(
            f"{suffix} is not a format this reads. Supported: "
            f"{', '.join(SUPPORTED)}, or paste the columns directly."
            + (" A photograph of a table is read by 'Photograph of a table' in "
               "Step 1, which proposes a grid for you to check first."
               if suffix in PICTURES else ""))

    frame = _tidy(frame)

    if frame.empty:
        raise UnreadableTable("the table has no rows")
    if len(frame.columns) < 2:
        raise UnreadableTable(
            "only one column was found — readings need at least time and "
            "drawdown. If the file is delimited by something unusual, save it "
            "as CSV.")
    return frame, notes


def _tidy(frame: pd.DataFrame) -> pd.DataFrame:
    """Make the frame safe for the inspector.

    Column detection matches patterns against the column *name*
    (`src/data/csv_inspector.py`), which assumes a string. Excel hands back
    integers for unnamed columns and datetimes for date headers, and a
    `TypeError` deep in inspection is an unhelpful way to learn that.
    """
    frame = frame.copy()
    frame.columns = [str(name).strip() for name in frame.columns]
    # Wholly empty rows and columns are what a spreadsheet's spacing leaves
    # behind, and they confuse the statistical layer of detection.
    frame = frame.dropna(axis="index", how="all").dropna(axis="columns", how="all")
    return frame.reset_index(drop=True)
