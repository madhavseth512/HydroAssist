"""Reading a table of readings, in whatever form it arrives.

The workspace accepted CSV and nothing else, which is not how field data
travels: an old export is tab-delimited text, a modern one is an Excel sheet,
and most people already have the numbers open in a spreadsheet and would rather
paste two columns than save a file first.

Only the reading changed. Everything downstream already took a DataFrame, so
these tests concentrate on the two things that can go wrong at the edge —
silently reading the wrong thing, and failing without saying what was tried.
"""

import io

import pandas as pd
import pytest

from src.services import table_reader
from src.services.table_reader import UnreadableTable, read_table

READINGS = pd.DataFrame({"time_s": [60, 120, 300, 600, 1800],
                         "drawdown_m": [0.12, 0.19, 0.31, 0.44, 0.71]})


def _excel(frame=READINGS, sheets=None) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        if sheets:
            for name, sheet_frame in sheets.items():
                sheet_frame.to_excel(writer, index=False, sheet_name=name)
        else:
            frame.to_excel(writer, index=False, sheet_name="Sheet1")
    return buffer.getvalue()


class TestTheFormatsPeopleActuallyHave:

    def test_csv_reads_as_before(self):
        frame, notes = read_table(READINGS.to_csv(index=False).encode(), "field.csv")

        assert list(frame.columns) == ["time_s", "drawdown_m"]
        assert len(frame) == 5
        assert notes == []

    @pytest.mark.parametrize("separator,name", [
        ("\t", "export.txt"), (";", "export.csv"), (",", "export.dat"),
        (r"\s+", "export.prn"),
    ])
    def test_the_delimiter_is_sniffed_not_assumed(self, separator, name):
        """A tab file and a semicolon file are both what somebody's software
        produced, and neither is worth a support question."""
        text = READINGS.to_csv(sep="\t" if separator == r"\s+" else separator,
                               index=False)
        frame, _ = read_table(text.encode(), name)

        assert list(frame.columns) == ["time_s", "drawdown_m"]
        assert len(frame) == 5

    def test_an_excel_sheet(self):
        frame, notes = read_table(_excel(), "field.xlsx")

        assert list(frame.columns) == ["time_s", "drawdown_m"]
        assert frame.time_s.tolist() == [60, 120, 300, 600, 1800]
        assert notes == []

    def test_pasted_text_needs_no_filename(self):
        frame, _ = read_table(READINGS.to_csv(index=False).encode())

        assert len(frame) == 5

    def test_pasted_columns_copied_from_a_spreadsheet(self):
        """What lands on the clipboard from Excel: tab-separated, no quoting."""
        pasted = "time_s\tdrawdown_m\n60\t0.12\n120\t0.19\n300\t0.31"

        frame, _ = read_table(pasted.encode(), table_reader.PASTED)

        assert len(frame) == 3
        assert frame.drawdown_m.tolist() == [0.12, 0.19, 0.31]

    def test_an_encoding_field_software_still_writes(self):
        """A degree sign in a header, written as latin-1, must not cost someone
        their upload."""
        text = "time_s,temperature °C\n60,21.5\n120,21.6\n"

        frame, _ = read_table(text.encode("latin-1"), "logger.csv")

        assert len(frame) == 2


class TestWhichSheetWasRead:

    def test_a_single_sheet_says_nothing(self):
        _, notes = read_table(_excel(), "field.xlsx")
        assert notes == []

    def test_several_sheets_say_which_one(self):
        """Silence here is dangerous: the wrong sheet fits the wrong data
        perfectly well, and the user would never know."""
        book = _excel(sheets={"Readings": READINGS, "Notes": READINGS})

        _, notes = read_table(book, "field.xlsx")

        assert notes and "Readings" in notes[0]
        assert "2" in notes[0]

    def test_a_cover_sheet_is_skipped_rather_than_fatal(self):
        """Real workbooks open with a title page or site notes. Taking the
        first sheet regardless would send people back to Excel to delete a tab
        before this would speak to them."""
        book = _excel(sheets={"Cover": pd.DataFrame({"Site": ["Gridley"]}),
                              "Test 1": READINGS})

        frame, notes = read_table(book, "field.xlsx")

        assert list(frame.columns) == ["time_s", "drawdown_m"]
        assert "Test 1" in notes[0]

    def test_a_workbook_with_no_table_anywhere_says_so(self):
        book = _excel(sheets={"Cover": pd.DataFrame({"Site": ["Gridley"]}),
                              "Notes": pd.DataFrame({"Note": ["windy"]})})

        with pytest.raises(UnreadableTable, match="two columns"):
            read_table(book, "field.xlsx")

    def test_a_named_sheet_is_used(self):
        book = _excel(sheets={"Notes": pd.DataFrame({"a": [1], "b": [2]}),
                              "Test 1": READINGS})

        frame, _ = read_table(book, "field.xlsx", sheet="Test 1")

        assert list(frame.columns) == ["time_s", "drawdown_m"]

    def test_a_sheet_that_is_not_there_lists_the_ones_that_are(self):
        book = _excel(sheets={"Notes": pd.DataFrame({"a": [1], "b": [2]}),
                              "Test 1": READINGS})

        with pytest.raises(UnreadableTable, match="Test 1"):
            read_table(book, "field.xlsx", sheet="Sheet9")


class TestWhatASpreadsheetDoesToAFrame:

    def test_column_labels_become_strings(self):
        """Column detection matches patterns against the name, which assumes a
        string. Excel hands back integers for unnamed columns and datetimes for
        date headers, and a TypeError deep inside inspection is an unhelpful way
        to find that out."""
        frame = pd.DataFrame({1: [60, 120], 2: [0.1, 0.2]})

        read, _ = read_table(_excel(frame), "field.xlsx")

        assert all(isinstance(name, str) for name in read.columns)

    def test_datetime_headers_survive(self):
        frame = pd.DataFrame({pd.Timestamp("2026-01-01"): [60, 120],
                              "drawdown_m": [0.1, 0.2]})

        read, _ = read_table(_excel(frame), "field.xlsx")

        assert all(isinstance(name, str) for name in read.columns)

    def test_the_blank_rows_a_spreadsheet_leaves_behind_are_dropped(self):
        padded = pd.concat([READINGS, pd.DataFrame({"time_s": [None],
                                                    "drawdown_m": [None]})])

        frame, _ = read_table(_excel(padded), "field.xlsx")

        assert len(frame) == 5

    def test_a_wholly_empty_column_is_dropped(self):
        wide = READINGS.assign(unused=[None] * 5)

        frame, _ = read_table(_excel(wide), "field.xlsx")

        assert "unused" not in frame.columns


class TestFailingUsefully:

    def test_an_empty_file(self):
        with pytest.raises(UnreadableTable, match="empty"):
            read_table(b"", "field.csv")

    def test_whitespace_only(self):
        with pytest.raises(UnreadableTable, match="empty"):
            read_table(b"   \n  \n", "field.csv")

    def test_a_format_it_does_not_read_names_the_ones_it_does(self):
        with pytest.raises(UnreadableTable) as raised:
            read_table(b"\x89PNG\r\n\x1a\n", "photo.png")

        message = str(raised.value)
        assert ".png" in message
        assert ".csv" in message and ".xlsx" in message
        assert "paste" in message

    def test_a_photograph_is_pointed_at_the_route_that_can_read_it(self):
        """A photograph is not unreadable, only unreadable *here* — it goes
        through OCR into a grid the user checks, never straight into a fit."""
        with pytest.raises(UnreadableTable) as raised:
            read_table(b"\xff\xd8\xff\xe0", "field-sheet.jpg")

        assert "Photograph of a table" in str(raised.value)

    def test_one_column_says_what_readings_need(self):
        """The commonest real failure: a file delimited by something the sniffer
        could not see, arriving as a single column of text."""
        with pytest.raises(UnreadableTable, match="time and drawdown"):
            read_table(b"time_and_drawdown\n60|0.12\n120|0.19\n", "odd.csv")

    def test_a_header_with_no_rows(self):
        with pytest.raises(UnreadableTable, match="no rows"):
            read_table(b"time_s,drawdown_m\n", "field.csv")

    def test_a_corrupt_workbook_is_reported_not_raised_raw(self):
        with pytest.raises(UnreadableTable):
            read_table(b"this is not a workbook", "field.xlsx")


class TestEveryRouteGivesTheSameNumbers:
    """A pasted table and an uploaded one must not diverge — the whole point of
    one reader behind every path."""

    def test_csv_text_and_excel_agree(self):
        from_csv, _ = read_table(READINGS.to_csv(index=False).encode(), "a.csv")
        from_text, _ = read_table(READINGS.to_csv(sep="\t", index=False).encode(),
                                  table_reader.PASTED)
        from_excel, _ = read_table(_excel(), "a.xlsx")

        pd.testing.assert_frame_equal(from_csv, from_text)
        pd.testing.assert_frame_equal(from_csv, from_excel, check_dtype=False)


class TestAFileThatBeginsWithNotes:
    """Reported from the deployed site, uploading this repository's own fixture:

        Failed to process the file: "[''] not in index"

    Its first four lines are `#` comments recording where the readings came from.
    They were read as the header row, split on the commas inside the prose, and
    the real header became a row of data — after which column detection named a
    column that did not exist and pandas raised from deep inside the formatter.

    A leading `#` marks a comment in nearly every ASCII data file a
    hydrogeologist meets, so this was not an unusual file.
    """

    GRIDLEY = "tests/fixtures/field_data/gridley_well1.csv"

    def test_the_readings_are_read(self):
        frame, _ = read_table(open(self.GRIDLEY, "rb").read(), "gridley1.csv")

        assert list(frame.columns) == ["time_min", "drawdown_ft"]
        assert len(frame) == 22
        assert frame.time_min.iloc[0] == 3

    def test_the_provenance_is_shown_rather_than_dropped(self):
        """Those comments say which test this is and what Walton published from
        it, which is worth putting in front of the user."""
        _, notes = read_table(open(self.GRIDLEY, "rb").read(), "gridley1.csv")

        assert any("Skipped 4 comment line" in note for note in notes)
        assert any("Bulletin 49" in note for note in notes)

    def test_every_field_dataset_in_the_repository_reads(self):
        """All three carry the same provenance header."""
        import glob

        for path in glob.glob("tests/fixtures/field_data/*.csv"):
            frame, _ = read_table(open(path, "rb").read(), path)
            assert len(frame.columns) == 2, path
            assert len(frame) > 5, path

    def test_a_comment_only_file_says_so(self):
        with pytest.raises(UnreadableTable, match="only comments"):
            read_table(b"# a note\n# another\n", "notes.csv")

    def test_blank_lines_before_the_header_are_skipped_too(self):
        frame, _ = read_table(b"\n\n# a note\n\ntime_s,drawdown_m\n60,0.12\n",
                              "field.csv")

        assert list(frame.columns) == ["time_s", "drawdown_m"]

    def test_a_hash_inside_a_well_name_is_not_a_comment(self):
        """`MW#3` is a well, not a remark. Truncating a line of readings at a
        `#` further down the file would lose data silently, so only the leading
        block is skipped."""
        frame, _ = read_table(b"time_s,MW#3,MW#4\n60,0.12,0.09\n", "wells.csv")

        assert list(frame.columns) == ["time_s", "MW#3", "MW#4"]
        assert frame["MW#3"].iloc[0] == 0.12

    def test_a_file_with_no_comments_gains_no_notes(self):
        _, notes = read_table(b"time_s,drawdown_m\n60,0.12\n", "field.csv")

        assert notes == []

    def test_pasted_text_may_also_start_with_a_comment(self):
        """Copying a fixture out of a text editor brings its header along."""
        frame, notes = read_table(
            b"# from the logger export\ntime_min\tdrawdown_m\n1\t0.34\n",
            table_reader.PASTED)

        assert list(frame.columns) == ["time_min", "drawdown_m"]
        assert notes


class TestWhenTheHeaderIsNotAHeader:
    """The other half of the same report: if detection ever does name a column
    the table has not got, the user must get a sentence rather than pandas'."""

    def test_the_failure_is_explained_in_words(self):
        import pandas as pd

        from src.data.csv_inspector import CSVInspector
        from src.data.formatter import DataFormatter

        frame = pd.DataFrame({"# Gridley, Illinois": ["Bulletin 49", "time_min"],
                              "unnamed": ["Table 7", "drawdown_ft"]})
        inspection = CSVInspector().inspect(frame)

        with pytest.raises(ValueError) as raised:
            DataFormatter().format(frame, inspection)

        message = str(raised.value)
        assert "does not look like a header" in message
        assert "# Gridley, Illinois" in message      # what it actually read
        assert "not in index" not in message
