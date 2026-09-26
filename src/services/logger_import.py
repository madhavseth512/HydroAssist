"""Readings as a pressure transducer writes them.

A logger does not record elapsed time and drawdown. It records a clock — a
timestamp per reading — and either a water level or a depth to water, in a file
that begins with a block of instrument metadata. Converting that by hand in
Excel, before HydroAssist will look at it, is the single largest barrier
between a practitioner and their own data.

Two conversions, both of which can be got wrong silently:

- **Clock to elapsed.** Seconds since the first reading, unless the file names
  the moment pumping started.
- **Water level to drawdown.** Drawdown is the *fall* from the pre-pumping
  level, so a depth-to-water column (which increases as the level falls) is
  already the right sign once the static level is subtracted, while a
  water-level column (which decreases) needs the subtraction the other way
  round. Getting this backwards yields a mirror-image curve that still fits,
  with a transmissivity that is wrong and no warning anywhere.

Because both can be wrong silently, nothing here is applied without saying so:
every conversion returns a note that ends up in front of the user.
"""

import io
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

# Solinst Levelogger (.lev), Van Essen Diver (.mon) and In-Situ exports all
# begin with a metadata block and mark where the readings start.
_DATA_MARKERS = ("[Data]", "[DATA]", "Date,Time", "Date;Time")

# Column names the instruments use, lower-cased for matching.
_TIMESTAMP_NAMES = ("datetime", "date_time", "timestamp", "date/time")
_DATE_NAMES = ("date",)
_TIME_NAMES = ("time",)
_DEPTH_NAMES = ("depth to water", "depth_to_water", "dtw", "depth")
_LEVEL_NAMES = ("level", "water level", "water_level", "head", "elevation",
                "pressure")

# Loggers write the unit into the column name ("LEVEL(ft)", "Depth m") or not
# at all. Naming the output column wrongly would hand feet to a calculator
# expecting metres and produce a transmissivity out by a factor of three.
_FEET = re.compile(r"(ft|feet|foot)|\(\s*ft\s*\)", re.I)
_METRES = re.compile(r"(m|metre|meter|metres|meters)|\(\s*m\s*\)", re.I)


@dataclass
class LoggerReadings:
    """What a logger file turned into, and what was done to it."""
    df: pd.DataFrame
    notes: List[str] = field(default_factory=list)
    static_level: Optional[float] = None
    instrument: str = ""


class NotALoggerFile(ValueError):
    """The file has no recognisable logger header."""


def looks_like_logger_export(raw: bytes) -> bool:
    """Whether this is worth handing to `read_logger_export`."""
    head = raw[:4000].decode("utf-8", errors="replace")
    return any(marker in head for marker in _DATA_MARKERS) or \
        bool(re.search(r"(Levelogger|Diver|In-?Situ|Solinst|Van Essen)", head, re.I))


def _split_header(text: str) -> Tuple[str, List[str]]:
    """The readings, and the instrument lines above them."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if any(marker in line for marker in _DATA_MARKERS):
            # "[Data]" is a marker; "Date,Time" is already the header row.
            start = index if line.strip().lower().startswith(("date", "date;")) \
                else index + 1
            return "\n".join(lines[start:]), lines[:index]
    raise NotALoggerFile("no data section was found in the file header")


def _instrument_of(header_lines: List[str]) -> str:
    for line in header_lines:
        match = re.search(r"(Levelogger|Barologger|Diver|In-?Situ|Solinst|"
                          r"Van Essen)[^,;\t]*", line, re.I)
        if match:
            return match.group(0).strip()
    return ""


def _find(columns, names) -> Optional[str]:
    lowered = {str(column).strip().lower(): column for column in columns}
    for name in names:
        for candidate, original in lowered.items():
            if candidate == name or candidate.startswith(name):
                return original
    return None


# An ISO date — 2026-03-01 — is unambiguous, and `dayfirst` must not be applied
# to it: pandas obligingly reads it as the 3rd of January, which shifts a test
# by months. Anything else (01/03/2026) genuinely is ambiguous, and loggers
# outside the United States write day first.
_ISO_DATE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}")


def _to_datetime(series: pd.Series) -> Tuple[pd.Series, bool]:
    """Timestamps, and whether the day-first reading was used."""
    text = series.astype(str)
    if text.str.match(_ISO_DATE).any():
        parsed = pd.to_datetime(text, errors="coerce")
        if not parsed.isna().all():
            return parsed, False

    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    return parsed, True


def _timestamps(frame: pd.DataFrame) -> Tuple[pd.Series, str]:
    """One clock column, however the instrument split it."""
    combined = _find(frame.columns, _TIMESTAMP_NAMES)
    if combined is not None:
        parsed, _ = _to_datetime(frame[combined])
        return parsed, str(combined)

    date = _find(frame.columns, _DATE_NAMES)
    time = _find(frame.columns, _TIME_NAMES)
    if date is not None and time is not None:
        joined = frame[date].astype(str).str.strip() + " " + \
            frame[time].astype(str).str.strip()
        parsed, _ = _to_datetime(joined)
        return parsed, f"{date} + {time}"

    raise NotALoggerFile(
        "no timestamp column was found — a logger export should carry a date "
        "and a time, or a combined date/time")


def read_logger_export(raw: bytes, static_level: Optional[float] = None,
                       pumping_started=None) -> LoggerReadings:
    """A logger file as time in seconds and drawdown in the file's own units.

    Args:
        raw: the file's bytes.
        static_level: the water level before pumping began. Without it the
            first reading is used, which is right only when recording started
            before the pump did — so it is reported either way.
        pumping_started: the moment pumping began, if it is not the first
            reading.

    Raises:
        NotALoggerFile: with what was missing.
    """
    text = raw.decode("utf-8", errors="replace")
    body, header_lines = _split_header(text)

    try:
        frame = pd.read_csv(io.StringIO(body), sep=None, engine="python")
    except Exception as error:
        raise NotALoggerFile(f"the data section could not be read: {error}")

    frame.columns = [str(column).strip() for column in frame.columns]
    notes = []
    instrument = _instrument_of(header_lines)
    if instrument:
        notes.append(f"Read a {instrument} export")

    stamps, stamp_source = _timestamps(frame)
    if stamps.isna().all():
        raise NotALoggerFile(f"the timestamps in {stamp_source!r} could not be read")

    origin = pd.to_datetime(pumping_started) if pumping_started is not None \
        else stamps.min()
    elapsed = (stamps - origin).dt.total_seconds()

    depth_column = _find(frame.columns, _DEPTH_NAMES)
    level_column = None if depth_column else _find(frame.columns, _LEVEL_NAMES)
    measurement = depth_column or level_column
    if measurement is None:
        raise NotALoggerFile(
            "no water level or depth-to-water column was found")

    values = pd.to_numeric(frame[measurement], errors="coerce")

    reference = static_level if static_level is not None else values.iloc[0]
    if static_level is None:
        notes.append(
            f"No static level was given, so the first reading "
            f"({reference:.3f}) was used as the pre-pumping level. If recording "
            f"began after pumping started, this understates every drawdown.")

    if depth_column:
        # Depth to water grows as the level falls, so drawdown is the increase.
        drawdown = values - reference
        notes.append(f"Drawdown from depth to water in {measurement!r}, "
                     f"measured against {reference:.3f}")
    else:
        # A water level falls, so drawdown is the decrease.
        drawdown = reference - values
        notes.append(f"Drawdown from water level in {measurement!r}, "
                     f"measured against {reference:.3f}")

    # The unit comes from the column name where the instrument wrote one; the
    # downstream formatter converts feet to metres, so the name must be right.
    unit_source = f"{measurement} {' '.join(header_lines[:6])}"
    if _FEET.search(str(measurement)) or (not _METRES.search(str(measurement))
                                          and _FEET.search(unit_source)):
        column, unit = "drawdown_ft", "feet"
    else:
        column, unit = "drawdown_m", "metres"
        if not _METRES.search(str(measurement)):
            notes.append(f"No unit was stated for {measurement!r}, so metres "
                         f"were assumed. Check the result if the logger "
                         f"recorded feet.")

    readings = pd.DataFrame({"time_s": elapsed, column: drawdown})
    readings = readings.dropna()
    before = len(readings)
    readings = readings[readings.time_s > 0]
    if before != len(readings):
        notes.append(f"{before - len(readings)} reading(s) at or before the "
                     f"start of pumping were left out")

    if readings.empty:
        raise NotALoggerFile("no readings remained after the start of pumping")

    notes.append(f"Elapsed time from {stamp_source!r}, "
                 f"{len(readings)} readings over "
                 f"{readings.time_s.max() / 3600:.1f} hours, drawdown in {unit}")

    return LoggerReadings(df=readings.reset_index(drop=True), notes=notes,
                          static_level=reference, instrument=instrument)
