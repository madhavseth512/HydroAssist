"""Reading a photographed field sheet, and refusing to trust it.

Field readings often exist only on paper: a printed table, or a hand-written
sheet on a clipboard, photographed on a phone. Typing forty rows twice — once on
site, once into a computer — is the last manual step between a practitioner and
their own data, and the tooling to remove it is already here:
`src/ingestion/processors/ocr.py` reads scanned papers for the knowledge base
with pypdfium2 and Tesseract.

What is different here is the standard of proof. For the knowledge base, a
mis-recognised word costs a slightly worse retrieval. For readings:

    A misread digit produces a plausible fit and a wrong transmissivity.

`3.5` read as `8.5` does not fail, does not warn, and cannot be detected
downstream. So this module only ever *proposes*. It returns every cell with the
confidence Tesseract assigned it, flags the doubtful ones, and leaves the
numbers outside the project until a person has looked at the grid. The API
carries that rule structurally: a confirmed grid re-enters through the same
route as typed text, so there is no path from an image to a fit that does not
pass through a person.

Two silent failures are guarded here rather than reported later:

- **A misread digit**, by Tesseract's own per-word confidence.
- **A misaligned column.** A row missing its middle cell shifts every later
  value one column left, which turns drawdowns into times. Cells are assigned to
  columns by where they sit on the page, not by their order in the row.
"""

import re
from dataclasses import dataclass, field
from statistics import median
from typing import Dict, List, Optional

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Tesseract's confidence is a percentage. Below about 80 it is guessing at one
# or more characters. A doubtful cell is flagged rather than dropped: it may
# well be right, and only the person holding the sheet can say.
UNCERTAIN_BELOW = 80.0

# A page where this share of cells is doubtful is not worth correcting one cell
# at a time, and saying so is kinder than a grid of yellow.
MOSTLY_UNCERTAIN = 0.4

# A reading is a plain decimal. A comma is a decimal point in most of the world,
# and a thousands separator never appears in a drawdown or an elapsed time.
_NUMBER = re.compile(r"^[-+]?\d{1,7}(?:[.,]\d{1,6})?$")

# The stray marks Tesseract leaves clinging to a cell of a ruled table: the rule
# itself read as a pipe, and brackets it invents around a lone digit — a printed
# "5" comes back as "5)" often enough to matter. Cleaning these lets the reading
# survive, and a cleaned cell is always flagged, because a number this had to
# tidy up is a number worth a second look.
#
# Deliberately not "." or ",": a leading one carries meaning. ".49" stripped to
# "49" would be a drawdown a hundred times too large, so it is refused instead.
_STRIP = " \t|:;'\"`()[]{}"

# Recognition costs both time and memory, and the deployed host has a tenth of a
# CPU and 512 MB — of which a loaded knowledge base and a conversation already
# hold 300 to 400. A modern phone photograph is 4,000 to 8,000 pixels across,
# which decodes to tens of megabytes before Tesseract has started; 2,000 is
# ample for digits a table's height, and the image is scaled rather than refused,
# since refusing would mean telling people to go and resize a photo.
MAX_DIMENSION = 2000


@dataclass
class Cell:
    """One number read from the image, with what Tesseract thought of it."""
    text: str
    value: Optional[float]
    confidence: float
    cleaned: bool = False

    @property
    def uncertain(self) -> bool:
        """Whether this cell is worth reading against the sheet first.

        Three ways to earn it: nothing numeric was recognised, Tesseract said it
        was unsure, or the token had to be tidied up before it was a number at
        all. The third matters because the tidying is a guess about what the mark
        beside the digit was.
        """
        return (self.value is None or self.confidence < UNCERTAIN_BELOW
                or self.cleaned)


@dataclass
class ReadTable:
    """What OCR made of a photograph, for a person to confirm or discard."""
    rows: List[List[Optional[Cell]]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    columns: int = 0

    @property
    def cells(self) -> List[Cell]:
        return [cell for row in self.rows for cell in row if cell is not None]

    @property
    def total_cells(self) -> int:
        return len(self.cells)

    @property
    def uncertain_cells(self) -> int:
        return sum(1 for cell in self.cells if cell.uncertain)

    @property
    def mostly_uncertain(self) -> bool:
        """Whether this reading is too doubtful to be worth correcting."""
        if not self.total_cells:
            return True
        return self.uncertain_cells / self.total_cells >= MOSTLY_UNCERTAIN

    def as_text(self) -> str:
        """The grid as tab-delimited text, which is what the ordinary reader
        takes. Gaps stay empty so a hole in the page stays a hole here."""
        return "\n".join(
            "\t".join("" if cell is None else cell.text for cell in row)
            for row in self.rows)


class OCRNotAvailable(RuntimeError):
    """Reading an image needs the optional OCR tooling, or the image failed."""


def ocr_available() -> bool:
    """Whether this machine can read an image at all.

    Separate from `processors.ocr.check_tooling`, which also requires pypdfium2
    for rendering PDF pages; a photograph needs only Pillow and Tesseract.
    """
    return not _missing_tooling()


def _missing_tooling() -> str:
    """What is missing, as something to show a user, or "" if nothing is."""
    try:
        import pytesseract
        from PIL import Image  # noqa: F401
    except ImportError:
        return ("Reading a photograph needs Pillow and pytesseract "
                "(pip install -r requirements.txt).")
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return ("Reading a photograph needs the Tesseract program itself, which "
                "is not a Python package — https://github.com/UB-Mannheim/"
                "tesseract/wiki on Windows, `apt install tesseract-ocr` on "
                "Debian or Ubuntu, `brew install tesseract` on macOS.")
    return ""


def _parse(text: str) -> Optional[float]:
    """A number, or None when the token is not one."""
    tidied = text.strip(_STRIP).replace(" ", "")
    if not _NUMBER.match(tidied):
        return None
    try:
        return float(tidied.replace(",", "."))
    except ValueError:
        return None


def _needed_tidying(text: str) -> bool:
    """Whether the token only became a number after stray marks came off."""
    return _parse(text) is not None and not _NUMBER.match(text.strip())


def read_image_table(raw: bytes, language: str = "eng") -> ReadTable:
    """Rows of numbers from a photograph of a table.

    Args:
        raw: the image's bytes — anything Pillow opens.
        language: the Tesseract model to use.

    Raises:
        OCRNotAvailable: with what to install, or why the image failed.
    """
    missing = _missing_tooling()
    if missing:
        raise OCRNotAvailable(missing)

    import io

    import pytesseract
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as error:
        raise OCRNotAvailable(f"that image could not be opened: {error}") from error

    scaled = ""
    grey = None
    try:
        if max(image.size) > MAX_DIMENSION:
            original = image.size
            image.thumbnail((MAX_DIMENSION, MAX_DIMENSION))
            scaled = (f"The image was {original[0]}×{original[1]}; it was scaled "
                      f"to {image.size[0]}×{image.size[1]} to read")
        # Tesseract binarises the image anyway, so colour buys nothing and costs
        # three bytes a pixel on a host with 512 MB. Held separately so both
        # bitmaps are released, not just whichever one the name ended up on.
        grey = image if image.mode in ("L", "1") else image.convert("L")
        data = pytesseract.image_to_data(
            grey, lang=language, output_type=pytesseract.Output.DICT)
    except Exception as error:
        raise OCRNotAvailable(
            f"Tesseract could not read the image: {error}") from error
    finally:
        if grey is not None and grey is not image:
            grey.close()
        image.close()

    table = rows_from_words(data)
    if scaled:
        table.notes.append(scaled)
    return table


# ── Turning recognised words into a grid ──────────────────────────────────────

@dataclass
class _Word:
    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int

    @property
    def centre(self) -> float:
        return self.left + self.width / 2


def _words_of(data: Dict) -> List[_Word]:
    """The recognised words, dropping Tesseract's empty placeholders."""
    keys = ("text", "conf", "left", "top", "width", "height")
    columns = [data.get(key, []) for key in keys]
    words = []
    for text, confidence, left, top, width, height in zip(*columns):
        if not str(text).strip():
            continue
        try:
            words.append(_Word(str(text), float(confidence), int(left),
                               int(top), int(width), int(height)))
        except (TypeError, ValueError):
            # A row of the dict that is not usable geometry is not a reading.
            continue
    return words


def _lines_of(words: List[_Word]) -> List[List[_Word]]:
    """Words grouped into printed lines by their vertical position.

    The tolerance comes from the text's own height rather than a fixed number of
    pixels, so a close-up photograph and a whole page both group correctly.
    """
    words = sorted(words, key=lambda word: (word.top, word.left))
    tolerance = max(4.0, 0.6 * median([word.height for word in words]))

    lines: List[List[_Word]] = [[words[0]]]
    for word in words[1:]:
        baseline = median([other.top for other in lines[-1]])
        if abs(word.top - baseline) <= tolerance:
            lines[-1].append(word)
        else:
            lines.append([word])
    return [sorted(line, key=lambda word: word.left) for line in lines]


def _column_centres(rows: List[List[_Word]]) -> List[float]:
    """Where the columns of the table sit on the page.

    Every kept cell's horizontal centre is gathered and split wherever the gap
    between consecutive centres exceeds a typical cell's width. This is what
    lets a row with a blank cell keep its remaining values in the right columns,
    instead of shifting them left and turning a drawdown into a time.
    """
    cells = [word for row in rows for word in row]
    centres = sorted(word.centre for word in cells)
    gap = max(8.0, 1.2 * median([word.width for word in cells]))

    clusters: List[List[float]] = [[centres[0]]]
    for centre in centres[1:]:
        if centre - clusters[-1][-1] <= gap:
            clusters[-1].append(centre)
        else:
            clusters.append([centre])
    return [sum(cluster) / len(cluster) for cluster in clusters]


def _place(row: List[_Word], centres: List[float]) -> List[Optional[Cell]]:
    """One line's words in their columns, with gaps left as gaps."""
    placed: List[Optional[Cell]] = [None] * len(centres)
    for word in row:
        index = min(range(len(centres)),
                    key=lambda position: abs(centres[position] - word.centre))
        cell = Cell(text=word.text.strip(_STRIP), value=_parse(word.text),
                    confidence=word.confidence,
                    cleaned=_needed_tidying(word.text))
        if placed[index] is None:
            placed[index] = cell
        else:
            # Two numbers competing for one column means the clustering does not
            # describe this page. Keeping both, side by side, is visible in the
            # grid; silently dropping one would not be.
            placed.append(cell)
    return placed


def rows_from_words(data: Dict) -> ReadTable:
    """A grid from Tesseract's word-level output.

    Split out from `read_image_table` so the layout logic — which is where a
    silent misalignment would come from — is testable without an image.
    """
    table = ReadTable()
    words = _words_of(data)
    if not words:
        table.notes.append("No text was recognised in that image.")
        return table

    numeric_rows, skipped, lone = [], 0, 0
    for line in _lines_of(words):
        numeric = [word for word in line if _parse(word.text) is not None]
        # A row of a table of readings is numbers. A line that is mostly words
        # is a heading, a note in the margin, or the top of the page.
        if len(numeric) >= 2 and len(numeric) >= len(line) / 2:
            numeric_rows.append(numeric)
        elif len(numeric) == 1:
            # One number on a line cannot establish itself as a row: a page
            # number looks exactly the same. It is counted separately because a
            # dropped reading and a dropped heading are different losses.
            lone += 1
        else:
            skipped += 1

    if not numeric_rows:
        table.notes.append(
            "No rows of numbers were found. A photograph of a printed table, "
            "taken square-on and in focus, reads best; a hand-written sheet "
            "often does not read at all.")
        return table

    centres = _column_centres(numeric_rows)
    table.rows = [_place(row, centres) for row in numeric_rows]
    table.columns = max(len(row) for row in table.rows)
    # A ragged grid means a cell was placed beyond the columns found, which is
    # the misalignment case; pad so the grid the user edits is rectangular.
    for row in table.rows:
        row.extend([None] * (table.columns - len(row)))

    table.notes.append(f"{len(table.rows)} rows and {table.columns} columns of "
                       f"numbers were read from the image")
    if skipped:
        table.notes.append(f"{skipped} line(s) were not rows of numbers and were "
                           f"left out — headings and marginalia usually")
    if lone:
        table.notes.append(f"{lone} line(s) held a single number and were left out, "
                           f"since one number could as easily be a page number. "
                           f"Check whether a reading was lost that way.")
    if table.uncertain_cells:
        table.notes.append(
            f"{table.uncertain_cells} of {table.total_cells} cells are uncertain "
            f"and are marked for checking")
    if any(cell is None for row in table.rows for cell in row):
        table.notes.append("Some cells were empty. Check that every number sits "
                           "in the column it does on the sheet.")
    if table.mostly_uncertain:
        table.notes.append(
            "Most of this reading is uncertain, so correcting it cell by cell is "
            "probably slower than typing the numbers or pasting them from a "
            "spreadsheet.")

    logger.info(f"OCR read {len(table.rows)}x{table.columns} cells, "
                f"{table.uncertain_cells}/{table.total_cells} uncertain")
    return table
