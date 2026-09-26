"""Readings photographed on a field sheet.

OCR is the one intake path that can be confidently wrong. A logger export either
parses or does not; a photograph always parses, and a misread digit gives a
plausible fit with a wrong transmissivity. So the tests here are less about
recognising more and more about the two ways a reading can be silently wrong:

- a digit Tesseract guessed at, which must reach the user marked as a guess;
- a cell placed in the wrong column, which turns a drawdown into a time and is
  invisible in the numbers themselves.

Most of these drive `rows_from_words` with Tesseract's own output shape, so the
layout logic is tested without an image or an installed program. The last class
runs the real thing, and is skipped where Tesseract is not installed.
"""

from pathlib import Path

import pytest

from src.services.table_ocr import (
    MOSTLY_UNCERTAIN, UNCERTAIN_BELOW, Cell, OCRNotAvailable, ocr_available,
    read_image_table, rows_from_words,
)

# A printed field sheet, rendered and committed so the whole path can be tested
# without depending on a font being installed on the machine.
FIELD_SHEET = "tests/fixtures/field_data/field_sheet.png"

# A cell of a printed table, at this many pixels across and down.
PITCH_X, PITCH_Y, WIDTH, HEIGHT = 120, 40, 60, 20


def word(text, column, line, confidence=96.0):
    return dict(text=text, conf=confidence, left=40 + column * PITCH_X,
                top=50 + line * PITCH_Y, width=WIDTH, height=HEIGHT)


def page(*words):
    """Tesseract's `image_to_data` dict, as it comes back."""
    keys = ("text", "conf", "left", "top", "width", "height")
    return {key: [entry[key] for entry in words] for key in keys}


def table(rows, confidence=96.0):
    """A clean grid: every cell present, every cell confident."""
    return page(*[word(text, column, line, confidence)
                  for line, row in enumerate(rows)
                  for column, text in enumerate(row)])


READINGS = [["1", "0.45"], ["2", "0.61"], ["5", "0.83"], ["10", "1.02"]]


class TestReadingTheRows:

    def test_a_clean_table_comes_back_whole(self):
        read = rows_from_words(table(READINGS))

        assert len(read.rows) == 4
        assert read.columns == 2
        assert [[cell.value for cell in row] for row in read.rows] == [
            [1.0, 0.45], [2.0, 0.61], [5.0, 0.83], [10.0, 1.02]]

    def test_a_heading_row_is_left_out(self):
        """"Time (min)  Drawdown (m)" is not a reading."""
        read = rows_from_words(table([["Time", "Drawdown"]] + READINGS))

        assert len(read.rows) == 4
        assert any("not rows of numbers" in note for note in read.notes)

    def test_a_note_in_the_margin_is_left_out(self):
        read = rows_from_words(table(READINGS + [["pump", "off"]]))

        assert len(read.rows) == 4

    def test_a_row_of_one_number_is_not_a_row(self):
        """A page number, or a single figure in a caption."""
        read = rows_from_words(page(word("1", 0, 0), word("0.45", 1, 0),
                                   word("2", 0, 1), word("0.61", 1, 1),
                                   word("7", 0, 2)))

        assert len(read.rows) == 2

    def test_a_line_holding_one_number_is_reported_as_left_out(self):
        """Dropping a page number and dropping a reading look identical from
        here, so the user is told a line went and can check which."""
        read = rows_from_words(page(word("1", 0, 0), word("0.45", 1, 0),
                                   word("2", 0, 1), word("0.61", 1, 1),
                                   word("7", 0, 2)))

        assert any("single number" in note for note in read.notes)

    def test_how_many_rows_and_columns_were_read_is_reported(self):
        read = rows_from_words(table(READINGS))

        assert any("4 rows and 2 columns" in note for note in read.notes)


class TestColumnsStayColumns:
    """The misalignment case, which the numbers alone cannot reveal.

    A reading missing from the middle of a row must not pull the values to its
    right one column left. Placement is by position on the page, not by order.
    """

    def test_a_gap_in_the_middle_stays_a_gap(self):
        read = rows_from_words(page(
            word("1", 0, 0), word("0.45", 1, 0), word("14.1", 2, 0),
            # The temperature was not recorded for this reading.
            word("2", 0, 1), word("0.61", 1, 1),
            word("5", 0, 2), word("0.83", 1, 2), word("14.3", 2, 2)))

        assert read.columns == 3
        assert [cell and cell.value for cell in read.rows[1]] == [2.0, 0.61, None]

    def test_a_gap_at_the_start_stays_a_gap(self):
        """The elapsed time was blank on the sheet; the drawdown must not slide
        into its column and be read as a time."""
        read = rows_from_words(page(
            word("1", 0, 0), word("0.45", 1, 0), word("14.1", 2, 0),
            word("0.61", 1, 1), word("14.2", 2, 1),
            word("5", 0, 2), word("0.83", 1, 2), word("14.3", 2, 2)))

        assert [cell and cell.value for cell in read.rows[1]] == [None, 0.61, 14.2]

    def test_an_empty_cell_is_called_out_for_checking(self):
        read = rows_from_words(page(
            word("1", 0, 0), word("0.45", 1, 0), word("14.1", 2, 0),
            word("0.61", 1, 1), word("14.2", 2, 1)))

        assert any("sits in the column" in note for note in read.notes)

    def test_a_slightly_skewed_photograph_still_lines_up(self):
        """A phone held by hand shifts each row a few pixels."""
        rows = []
        for line, (time, drawdown) in enumerate(READINGS):
            skew = line * 4
            rows.append(dict(word(time, 0, line), left=40 + skew))
            rows.append(dict(word(drawdown, 1, line), left=40 + PITCH_X + skew))

        read = rows_from_words(page(*rows))

        assert read.columns == 2
        assert [cell.value for cell in read.rows[3]] == [10.0, 1.02]

    def test_the_grid_is_rectangular(self):
        """Whatever the page did, the grid a user edits has equal rows."""
        read = rows_from_words(page(
            word("1", 0, 0), word("0.45", 1, 0), word("14.1", 2, 0),
            word("2", 0, 1), word("0.61", 1, 1)))

        assert len({len(row) for row in read.rows}) == 1


class TestFlaggingWhatItGuessedAt:

    def test_a_low_confidence_cell_is_uncertain(self):
        read = rows_from_words(page(
            word("1", 0, 0), word("0.45", 1, 0),
            word("2", 0, 1), word("0.61", 1, 1, confidence=41.0)))

        assert read.rows[1][1].uncertain is True
        assert read.rows[1][0].uncertain is False
        assert read.uncertain_cells == 1
        assert read.total_cells == 4

    def test_the_count_of_uncertain_cells_reaches_the_user(self):
        read = rows_from_words(page(
            word("1", 0, 0), word("0.45", 1, 0, confidence=50.0),
            word("2", 0, 1), word("0.61", 1, 1)))

        assert any("1 of 4 cells are uncertain" in note for note in read.notes)

    def test_a_confident_table_says_nothing_about_uncertainty(self):
        read = rows_from_words(table(READINGS))

        assert read.uncertain_cells == 0
        assert not any("uncertain" in note for note in read.notes)

    def test_a_mostly_doubtful_page_says_typing_would_be_quicker(self):
        """Correcting two thirds of a grid by hand is worse than typing it."""
        read = rows_from_words(table(READINGS, confidence=30.0))

        assert read.mostly_uncertain is True
        assert any("typing the numbers" in note for note in read.notes)

    def test_the_threshold_is_where_it_says_it_is(self):
        assert Cell("1", 1.0, UNCERTAIN_BELOW).uncertain is False
        assert Cell("1", 1.0, UNCERTAIN_BELOW - 0.1).uncertain is True
        assert 0.0 < MOSTLY_UNCERTAIN < 1.0


class TestWhatCountsAsANumber:

    @pytest.mark.parametrize("text,expected", [
        ("0.45", 0.45),
        ("0,45", 0.45),       # a decimal comma, as most of the world writes it
        ("-0.12", -0.12),     # a recovering water level
        ("|1.5", 1.5),        # a ruled table leaves its rule on the cell
        ("10", 10.0),
    ])
    def test_these_are_readings(self, text, expected):
        read = rows_from_words(page(word(text, 0, 0), word("1", 1, 0)))

        assert read.rows[0][0].value == pytest.approx(expected)

    @pytest.mark.parametrize("text", ["1.2.3", "O.45", "12:30", "n/a", "3-4"])
    def test_these_are_not(self, text):
        """`O.45` matters: a capital O read for a zero must not become 0.45."""
        read = rows_from_words(page(word(text, 0, 0), word("1", 1, 0),
                                   word("2", 2, 0)))

        assert all(cell is None or cell.text != text for cell in read.rows[0])


class TestFailingUsefully:

    def test_an_image_with_no_text(self):
        read = rows_from_words(page())

        assert read.rows == []
        assert any("No text was recognised" in note for note in read.notes)

    def test_a_page_of_prose(self):
        read = rows_from_words(table([["The", "pumping", "test"],
                                      ["began", "at", "nine"]]))

        assert read.rows == []
        assert any("square-on" in note for note in read.notes)

    def test_an_empty_reading_is_treated_as_wholly_uncertain(self):
        """Nothing read is not the same as nothing wrong."""
        assert rows_from_words(page()).mostly_uncertain is True

    def test_a_file_that_is_not_an_image(self):
        if not ocr_available():
            pytest.skip("Tesseract is not installed on this machine")

        with pytest.raises(OCRNotAvailable, match="could not be opened"):
            read_image_table(b"time_s,drawdown_m\n60,0.12\n")


class TestHandingOnToTheOrdinaryReader:
    """A confirmed grid must arrive as readings by the same route as typed text,
    so an image can never reach a calculator on its own."""

    def test_the_grid_becomes_tab_delimited_text(self):
        read = rows_from_words(table(READINGS))

        assert read.as_text() == ("1\t0.45\n2\t0.61\n5\t0.83\n10\t1.02")

    def test_a_gap_stays_empty_rather_than_becoming_a_zero(self):
        """An empty cell reaching a calculator as 0 would be a reading nobody
        took, at a time nobody measured."""
        read = rows_from_words(page(
            word("1", 0, 0), word("0.45", 1, 0), word("14.1", 2, 0),
            word("0.61", 1, 1), word("14.2", 2, 1)))

        assert read.as_text().splitlines()[1] == "\t0.61\t14.2"

    def test_headed_text_reads_as_readings(self):
        from src.services.table_reader import read_table

        read = rows_from_words(table(READINGS))
        frame, _ = read_table(
            ("time_min\tdrawdown_m\n" + read.as_text()).encode("utf-8"),
            "pasted.txt")

        assert list(frame.columns) == ["time_min", "drawdown_m"]
        assert frame.drawdown_m.tolist() == [0.45, 0.61, 0.83, 1.02]


class TestARealPhotograph:
    """The whole path, on a printed sheet read by Tesseract itself.

    The fixture is a rendered field sheet — a title, a heading row and six
    readings — committed so this test does not depend on a font being installed.
    Skipped where Tesseract is absent, since it is a program, not a package.
    """

    @pytest.fixture(scope="class")
    def photograph(self):
        if not ocr_available():
            pytest.skip("Tesseract is not installed on this machine")
        return Path(FIELD_SHEET).read_bytes()

    def test_the_readings_are_recovered(self, photograph):
        read = read_image_table(photograph)

        assert [[cell and cell.value for cell in row] for row in read.rows] == [
            [1.0, 0.22], [2.0, 0.33], [3.0, 0.40], [5.0, 0.49], [7.0, 0.55],
            [10.0, 0.61], [15.0, 0.68], [20.0, 0.74], [30.0, 0.81],
            [45.0, 0.89], [60.0, 0.94], [90.0, 1.01], [120.0, 1.07]]

    def test_neither_the_title_nor_the_heading_is_taken_for_a_reading(
            self, photograph):
        read = read_image_table(photograph)

        assert len(read.rows) == 13
        assert any("not rows of numbers" in note for note in read.notes)

    def test_printed_text_is_read_confidently(self, photograph):
        """Printed and square-on is the case OCR is good at, so nearly every
        cell should come back trusted; doubt everywhere would mean the thresholds
        are set for the wrong world."""
        read = read_image_table(photograph)

        assert read.uncertain_cells <= 1
        assert read.mostly_uncertain is False

    def test_the_one_cell_it_stumbled_on_is_flagged_rather_than_lost(
            self, photograph):
        """On this sheet Tesseract reads the printed "5" as "5)" at 71% — which
        is the whole case for tidying the token and marking the cell instead of
        rejecting it. Rejecting it dropped the entire reading, because a line
        with one number on it is not a row.
        """
        read = read_image_table(photograph)
        flagged = [cell for row in read.rows for cell in row
                   if cell is not None and cell.uncertain]

        assert [cell.value for cell in flagged] == [5.0]
        assert any("marked for checking" in note for note in read.notes)

    def test_a_colour_photograph_at_phone_resolution_is_scaled_and_read(
            self, photograph):
        """What actually arrives: several thousand pixels across, in colour.

        Decoding that costs tens of megabytes before Tesseract starts, on a host
        with 512 MB of which the knowledge base already holds most. It is scaled
        and reduced to grey rather than refused, and the scaling is reported.
        """
        import io

        from PIL import Image

        sheet = Image.open(io.BytesIO(photograph)).convert("RGB")
        big = sheet.resize((sheet.width * 5, sheet.height * 5), Image.LANCZOS)
        buffer = io.BytesIO()
        big.save(buffer, format="JPEG", quality=88)

        read = read_image_table(buffer.getvalue())

        assert len(read.rows) == 13
        assert [cell.value for cell in read.rows[0]] == [1.0, 0.22]
        assert any("it was scaled to" in note for note in read.notes)

    def test_the_grid_is_offered_as_text_the_ordinary_reader_takes(
            self, photograph):
        from src.services.table_reader import read_table

        read = read_image_table(photograph)
        frame, _ = read_table(
            ("time_min\tdrawdown_m\n" + read.as_text()).encode("utf-8"),
            "pasted.txt")

        assert len(frame) == 13
        assert frame.drawdown_m.iloc[-1] == 1.07
