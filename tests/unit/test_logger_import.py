"""Readings as a pressure transducer writes them.

A logger records a clock and a water level, not elapsed seconds and drawdown,
and it puts a block of instrument metadata above the numbers. Converting that by
hand in Excel before HydroAssist will look at it is the largest barrier between
a practitioner and their own data.

Two of these conversions can be wrong *silently*, which is what most of these
tests are about:

- A water level falls as drawdown grows, while a depth to water rises. Getting
  the sign backwards yields a mirror-image curve that still fits, with a
  transmissivity that is wrong and no warning anywhere.
- A level in feet read as metres is out by a factor of three, and nothing later
  in the pipeline can tell.
"""

import pandas as pd
import pytest

from src.services.logger_import import (
    NotALoggerFile, looks_like_logger_export, read_logger_export,
)

SOLINST = b"""Serial_number: 2048322
Instrument type: Levelogger Edge M10
Location: PW-1
[Data]
Date,Time,LEVEL(m),TEMPERATURE
2026-03-01,09:00:00,12.480,14.2
2026-03-01,09:01:00,12.100,14.2
2026-03-01,09:05:00,11.640,14.3
2026-03-01,09:30:00,10.980,14.3
2026-03-01,10:00:00,10.640,14.4
"""

DIVER = b"""Data file for DataLogger.
[Logger settings]
Instrument type = Van Essen Diver
[Data]
Date/time,Depth to water (m),Temperature
2026-03-01 09:00:00,3.520,14.1
2026-03-01 09:02:00,3.900,14.1
2026-03-01 09:20:00,4.560,14.2
2026-03-01 10:00:00,5.360,14.2
"""


class TestRecognisingOne:

    @pytest.mark.parametrize("raw", [SOLINST, DIVER])
    def test_a_logger_export_is_recognised_by_its_header(self, raw):
        assert looks_like_logger_export(raw) is True

    def test_an_ordinary_csv_is_not(self):
        assert looks_like_logger_export(
            b"time_s,drawdown_m\n60,0.12\n120,0.19\n") is False

    def test_recognition_does_not_depend_on_the_extension(self):
        """In-Situ writes this shape into a file named .csv."""
        assert looks_like_logger_export(SOLINST) is True


class TestTheClock:

    def test_timestamps_become_seconds_from_the_first_reading(self):
        readings = read_logger_export(SOLINST)

        assert readings.df.time_s.tolist() == [60.0, 300.0, 1800.0, 3600.0]

    def test_a_split_date_and_time_are_joined(self):
        readings = read_logger_export(SOLINST)
        assert "Date + Time" in " ".join(readings.notes)

    def test_a_combined_column_works_too(self):
        readings = read_logger_export(DIVER)
        assert readings.df.time_s.tolist() == [120.0, 1200.0, 3600.0]

    def test_pumping_can_start_later_than_recording(self):
        """Loggers are usually started before the pump, which is why the origin
        can be given rather than assumed."""
        readings = read_logger_export(SOLINST,
                                      pumping_started="2026-03-01 09:05:00")

        assert readings.df.time_s.tolist() == [1500.0, 3300.0]

    def test_readings_before_pumping_are_left_out_with_a_count(self):
        readings = read_logger_export(SOLINST,
                                      pumping_started="2026-03-01 09:05:00")

        assert any("left out" in note for note in readings.notes)

    def test_unreadable_timestamps_say_so(self):
        broken = b"[Data]\nDate,Time,LEVEL\nnot a date,nor a time,1.0\n"

        with pytest.raises(NotALoggerFile, match="timestamps"):
            read_logger_export(broken)


class TestTheSignOfDrawdown:
    """The conversion that fails silently, and the reason this module exists."""

    def test_a_falling_water_level_is_a_growing_drawdown(self):
        readings = read_logger_export(SOLINST)

        assert readings.df.drawdown_m.tolist() == pytest.approx(
            [0.38, 0.84, 1.50, 1.84])
        assert readings.df.drawdown_m.is_monotonic_increasing

    def test_a_rising_depth_to_water_is_also_a_growing_drawdown(self):
        """Opposite sign in the file, same sign in the result."""
        readings = read_logger_export(DIVER)

        assert readings.df.drawdown_m.tolist() == pytest.approx([0.38, 1.04, 1.84])
        assert readings.df.drawdown_m.is_monotonic_increasing

    def test_which_column_was_used_and_against_what_is_reported(self):
        readings = read_logger_export(SOLINST)
        notes = " ".join(readings.notes)

        assert "LEVEL(m)" in notes and "12.480" in notes

    def test_a_given_static_level_is_used_instead_of_the_first_reading(self):
        readings = read_logger_export(SOLINST, static_level=12.600)

        assert readings.static_level == 12.600
        assert readings.df.drawdown_m.iloc[0] == pytest.approx(0.50)

    def test_assuming_the_first_reading_is_stated_as_an_assumption(self):
        """It is right only when recording began before pumping, and the user is
        the only one who knows whether it did."""
        readings = read_logger_export(SOLINST)

        assert any("No static level was given" in note for note in readings.notes)
        assert any("understates every drawdown" in note for note in readings.notes)


class TestUnits:

    def test_feet_are_named_feet_so_they_are_converted_later(self):
        raw = SOLINST.replace(b"LEVEL(m)", b"LEVEL(ft)")

        readings = read_logger_export(raw)

        assert "drawdown_ft" in readings.df.columns

    def test_metres_are_named_metres(self):
        assert "drawdown_m" in read_logger_export(SOLINST).df.columns

    def test_an_unstated_unit_is_assumed_and_the_assumption_is_shown(self):
        raw = SOLINST.replace(b"LEVEL(m)", b"LEVEL")

        readings = read_logger_export(raw)

        assert "drawdown_m" in readings.df.columns
        assert any("metres were assumed" in note for note in readings.notes)


class TestFailingUsefully:

    def test_a_file_with_no_data_section(self):
        with pytest.raises(NotALoggerFile, match="data section"):
            read_logger_export(b"Serial_number: 1\nInstrument type: Diver\n")

    def test_a_file_with_no_level_column(self):
        raw = b"[Data]\nDate,Time,TEMPERATURE\n2026-03-01,09:00:00,14.2\n"

        with pytest.raises(NotALoggerFile, match="water level"):
            read_logger_export(raw)

    def test_a_file_whose_readings_all_precede_pumping(self):
        with pytest.raises(NotALoggerFile, match="no readings remained"):
            read_logger_export(SOLINST, pumping_started="2026-03-02 09:00:00")

    def test_the_instrument_is_named_when_the_header_says(self):
        assert "Levelogger" in read_logger_export(SOLINST).instrument
        assert "Diver" in read_logger_export(DIVER).instrument


class TestThroughTheOrdinaryUploadPath:
    """One reader behind every route, so a logger file needs no special button."""

    def test_a_lev_file_arrives_as_readings(self):
        from src.services.table_reader import read_table

        frame, notes = read_table(SOLINST, "PW-1.lev")

        assert list(frame.columns) == ["time_s", "drawdown_m"]
        assert len(frame) == 4
        assert any("Levelogger" in note for note in notes)

    def test_a_logger_export_named_csv_is_still_recognised(self):
        from src.services.table_reader import read_table

        frame, _ = read_table(DIVER, "download.csv")

        assert list(frame.columns) == ["time_s", "drawdown_m"]

    def test_an_ordinary_csv_is_not_treated_as_a_logger_file(self):
        from src.services.table_reader import read_table

        frame, notes = read_table(b"time_min,drawdown_ft\n1,0.3\n5,0.7\n", "a.csv")

        assert list(frame.columns) == ["time_min", "drawdown_ft"]
        assert notes == []

    def test_a_broken_logger_file_reports_in_the_callers_language(self):
        from src.services.table_reader import UnreadableTable, read_table

        with pytest.raises(UnreadableTable):
            read_table(b"Instrument type: Diver\nno data here\n", "broken.lev")


class TestHowDatesAreRead:
    """The bug this class exists for: `dayfirst=True` applied to an ISO date.

    pandas obligingly read "2026-03-01" as the 3rd of January. Differences
    within one day survive that unharmed, which is why the first version of
    these tests passed — but an origin given from outside, or a test spanning
    midnight, is silently shifted by months.
    """

    def test_an_iso_date_is_read_as_iso(self):
        raw = (b"[Data]\nDate,Time,LEVEL(m)\n"
               b"2026-03-01,09:00:00,12.0\n2026-03-01,09:10:00,11.0\n")

        readings = read_logger_export(raw, pumping_started="2026-03-01 09:00:00")

        assert readings.df.time_s.tolist() == [600.0]

    def test_a_day_first_date_is_read_day_first(self):
        """01/03/2026 is genuinely ambiguous, and loggers outside the United
        States write the day first."""
        raw = (b"[Data]\nDate/time,LEVEL(m)\n"
               b"01/03/2026 09:00:00,12.0\n01/03/2026 09:10:00,11.0\n")

        readings = read_logger_export(raw, pumping_started="2026-03-01 09:00:00")

        assert readings.df.time_s.tolist() == [600.0]

    def test_a_test_spanning_midnight_keeps_its_hours(self):
        """Where a flipped month shows up as nonsense rather than as a shift."""
        raw = (b"[Data]\nDate,Time,LEVEL(m)\n"
               b"2026-03-01,23:30:00,12.0\n2026-03-02,00:30:00,11.0\n"
               b"2026-03-02,01:30:00,10.5\n")

        readings = read_logger_export(raw)

        assert readings.df.time_s.tolist() == [3600.0, 7200.0]
