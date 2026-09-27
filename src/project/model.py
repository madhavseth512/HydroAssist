"""Project model and file format.

AQTESOLV organises work around a saved, named, reopenable data set holding the
aquifer data, the wells, the readings and the analyses run against them. Before
this module HydroAssist had no such concept: every session started empty, nothing
survived a page refresh, and analysing a second test meant re-entering everything.
That made it a demonstration rather than a tool a hydrogeologist could do a job in.

A Project holds client-facing metadata and one or more PumpingTests. Each test
holds its readings, its setup, and every analysis run against it, so methods can
be compared side by side instead of being recomputed from memory.

Storage is a JSON file the user downloads and re-opens. That was a deliberate
choice for the free-tier host: it needs no accounts, no database and no persistent
disk, and it survives the host sleeping. It is also portable — a project can be
emailed to a colleague or attached to a report.

Two correctness rules are built in rather than left to the UI:

  * Analyses are only compared when they were run on the same readings. Each one
    records a fingerprint of the data it saw; replacing a test's readings marks
    older analyses stale instead of silently comparing models fitted to different
    data.
  * Models are ranked by AICc, never by R². R² improves with every added
    parameter, so ranking on it would always favour the most complex method.
"""

import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

SCHEMA = "hydroassist.project"
# 2: TestSetup gains test_type, t_p_s and recovery_clock for recovery tests.
# Version-1 files open unchanged; the new fields default to a constant-rate test.
SCHEMA_VERSION = 2
FILE_SUFFIX = ".hydroassist.json"

# A project file is parsed from an upload, so bound what we are willing to read.
MAX_FILE_BYTES = 20 * 1024 * 1024


class ProjectFileError(ValueError):
    """A project file that cannot be opened, with a message the user can act on."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _finite_or_none(value):
    """JSON has no NaN or infinity; store them as null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sanitise(obj):
    """Recursively make a structure strictly JSON-serialisable."""
    if isinstance(obj, dict):
        return {str(k): _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return _finite_or_none(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_sanitise(v) for v in obj.tolist()]
    if isinstance(obj, float):
        return _finite_or_none(obj)
    return obj


def _estimates(result: dict, name: str) -> bool:
    params = result.get("parameters")
    return not params or any(p.get("name") == name for p in params)


# ── Building blocks ───────────────────────────────────────────────────────────

@dataclass
class ProjectInfo:
    """The metadata a client report carries."""

    title: str = "Untitled project"
    client: str = ""
    project_number: str = ""
    location: str = ""
    test_date: str = ""          # ISO date, as entered
    analyst: str = ""
    notes: str = ""


@dataclass
class Readings:
    """Formatted, SI time-drawdown data for one test."""

    time_s: List[float]
    drawdown_m: List[float]
    well_id: Optional[List[str]] = None
    source_name: str = ""
    conversions: List[str] = field(default_factory=list)

    def __post_init__(self):
        if len(self.time_s) != len(self.drawdown_m):
            raise ValueError("time and drawdown must have the same number of values.")
        if self.well_id is not None and len(self.well_id) != len(self.time_s):
            raise ValueError("well_id must have one entry per reading.")

    @classmethod
    def from_frame(cls, df: pd.DataFrame, source_name: str = "",
                   conversions: Optional[List[str]] = None) -> "Readings":
        return cls(
            time_s=[float(v) for v in df["time_s"]],
            drawdown_m=[float(v) for v in df["drawdown_m"]],
            well_id=([str(v) for v in df["well_id"]]
                     if "well_id" in df.columns else None),
            source_name=source_name,
            conversions=list(conversions or []),
        )

    def to_frame(self) -> pd.DataFrame:
        data = {"time_s": self.time_s, "drawdown_m": self.drawdown_m}
        if self.well_id is not None:
            data["well_id"] = self.well_id
        return pd.DataFrame(data)

    @property
    def fingerprint(self) -> str:
        """Identifies this exact data, so analyses on different data are never
        compared with each other."""
        payload = json.dumps([self.time_s, self.drawdown_m, self.well_id])
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def __len__(self) -> int:
        return len(self.time_s)


@dataclass
class TestSetup:
    """The test parameters an analysis is run with. Stored in SI, with the unit
    the user entered Q in kept for display, so a reopened project reads back the
    way it was typed."""

    # The name starts with "Test", so without this pytest tries to collect the
    # class as a test suite in any module that imports it.
    __test__ = False

    Q_m3s: Optional[float] = None
    Q_entered: Optional[float] = None
    Q_unit: str = "m³/day"
    r_m: float = 50.0
    r_w_m: float = 0.10
    r_c_m: float = 0.10

    # Recovery tests: how long the well pumped before shut-down, and whether the
    # time column counts from the start or the end of pumping.
    test_type: str = "constant_rate"
    t_p_s: Optional[float] = None
    recovery_clock: str = "since_stop"


@dataclass
class AnalysisRecord:
    """One completed analysis, stored with everything needed to trust it later."""

    method_id: str
    method_label: str
    setup: TestSetup
    result: dict                   # CalculationResult.to_state_dict()
    readings_fingerprint: str
    id: str = field(default_factory=_new_id)
    created_at: str = field(default_factory=_now)
    note: str = ""

    @property
    def success(self) -> bool:
        return bool(self.result.get("success"))

    @property
    def gof(self) -> dict:
        return self.result.get("gof") or {}

    @property
    def aicc(self) -> Optional[float]:
        return self.gof.get("aicc")

    @property
    def n_params(self) -> Optional[int]:
        return self.gof.get("n_params")


# How many runs a test keeps. A working session tries a handful of methods; the
# cap is only there so a project file cannot grow without bound.
RUNS_KEPT = 20


@dataclass
class RunRecord:
    """A fit that was run against this test — saved for comparison or not.

    `analyses` holds the ones deliberately saved, with everything needed to trust
    them later: the full result, the setup, the readings fingerprint. This is the
    lighter record of *what was run*, and it exists because those were two
    different questions and only one of them had an answer.

    Reported twice from the deployed site: two tests in the project panel, and
    clicking either showed no analysis at all. The readings were kept per test;
    the fit was kept once per session and discarded the moment another test was
    opened, so an afternoon's work left nothing behind.

    Deliberately not a whole `CalculationResult`: the fitted curve and the
    residuals are derived from the readings, which are here, and storing them per
    run would multiply the file size for numbers a re-run reproduces exactly.
    """

    method_label: str
    method_id: str = ""
    success: bool = True
    T_day: Optional[float] = None
    S: Optional[float] = None
    r_squared: Optional[float] = None
    rmse: Optional[float] = None
    n_points: Optional[int] = None
    readings_fingerprint: str = ""
    error_message: str = ""
    created_at: str = field(default_factory=_now)


@dataclass
class PumpingTest:
    """One test: its wells, readings, setup and the analyses run against it."""

    name: str
    id: str = field(default_factory=_new_id)
    pumping_well: str = ""
    observation_well: str = ""
    readings: Optional[Readings] = None
    setup: TestSetup = field(default_factory=TestSetup)
    analyses: List[AnalysisRecord] = field(default_factory=list)
    runs: List[RunRecord] = field(default_factory=list)

    def set_readings(self, readings: Readings) -> None:
        """Replace the data. Earlier analyses are kept but become stale, because
        they describe data this test no longer holds."""
        self.readings = readings

    def record_run(self, result, method_label: str, method_id: str = "",
                   n_points: Optional[int] = None) -> RunRecord:
        """Note that this fit was run, whether or not it is saved.

        A method re-run on the same readings replaces its earlier entry: the later
        fit is the one that stands, and two rows for one method invite a
        comparison that is not one. Re-run after the readings change, it does not
        replace anything — those are different results about different data.
        """
        payload = (result.to_state_dict() if hasattr(result, "to_state_dict")
                   else dict(result or {}))
        fingerprint = self.readings.fingerprint if self.readings else ""
        run = RunRecord(
            method_label=method_label,
            method_id=method_id or payload.get("method_id", ""),
            success=bool(payload.get("success")),
            T_day=payload.get("T_day"),
            S=payload.get("S"),
            r_squared=payload.get("r_squared"),
            rmse=payload.get("rmse"),
            n_points=n_points if n_points is not None else (
                len(self.readings) if self.readings else None),
            readings_fingerprint=fingerprint,
            error_message=payload.get("error_message", "") or "",
        )
        self.runs = [existing for existing in self.runs
                     if not (existing.method_label == run.method_label
                             and existing.readings_fingerprint == fingerprint)]
        self.runs.append(run)
        self.runs = self.runs[-RUNS_KEPT:]
        return run

    def current_runs(self) -> List[RunRecord]:
        """The runs that describe the readings this test holds now.

        A run against replaced data is kept in the file — it happened — but it is
        not offered for comparison, for the same reason a stale analysis is not.
        """
        if self.readings is None:
            return []
        return [run for run in self.runs
                if run.readings_fingerprint == self.readings.fingerprint]

    def add_analysis(self, result, method_label: str, method_id: str = "",
                     setup: Optional[TestSetup] = None,
                     note: str = "") -> AnalysisRecord:
        """Record a completed analysis against this test's current readings."""
        if self.readings is None:
            raise ValueError("Load readings into this test before saving an analysis.")

        payload = result.to_state_dict() if hasattr(result, "to_state_dict") else dict(result)
        record = AnalysisRecord(
            method_id=method_id or payload.get("method_id", ""),
            method_label=method_label,
            setup=setup if setup is not None else TestSetup(**vars(self.setup)),
            result=_sanitise(payload),
            readings_fingerprint=self.readings.fingerprint,
            note=note,
        )
        self.analyses.append(record)
        return record

    def is_stale(self, record: AnalysisRecord) -> bool:
        return (self.readings is None
                or record.readings_fingerprint != self.readings.fingerprint)

    def remove_analysis(self, analysis_id: str) -> None:
        self.analyses = [a for a in self.analyses if a.id != analysis_id]

    def comparison(self) -> List[dict]:
        """Successful, current analyses ranked by AICc, best first.

        Each row carries ΔAICc against the best model and its Akaike weight — the
        relative likelihood that it is the best of the set. R² is reported but
        never used for ranking, because it cannot penalise extra parameters.
        """
        current = [a for a in self.analyses if a.success and not self.is_stale(a)]
        scored = [a for a in current if a.aicc is not None]
        best = min((a.aicc for a in scored), default=None)

        weights = {}
        if scored:
            raw = {a.id: math.exp(-0.5 * (a.aicc - best)) for a in scored}
            total = sum(raw.values())
            weights = {k: v / total for k, v in raw.items()}

        rows = []
        for a in current:
            rows.append({
                "analysis_id": a.id,
                "method": a.method_label,
                "n_params": a.n_params,
                "T_m2_per_day": a.result.get("T_day"),
                # None rather than the 0.0 placeholder when the method did not
                # estimate S — recovery gives only S/S'.
                "S": a.result.get("S") if _estimates(a.result, "S") else None,
                "r_squared": a.result.get("r_squared"),
                "aicc": a.aicc,
                "delta_aicc": (a.aicc - best) if a.aicc is not None else None,
                "akaike_weight": weights.get(a.id),
            })
        rows.sort(key=lambda r: (r["aicc"] is None,
                                 r["aicc"] if r["aicc"] is not None else 0.0))
        return rows


# ── Project ───────────────────────────────────────────────────────────────────

@dataclass
class Project:
    info: ProjectInfo = field(default_factory=ProjectInfo)
    tests: List[PumpingTest] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    modified_at: str = field(default_factory=_now)

    def add_test(self, name: str = "") -> PumpingTest:
        test = PumpingTest(name=name or f"Test {len(self.tests) + 1}")
        self.tests.append(test)
        self.touch()
        return test

    def get_test(self, test_id: str) -> PumpingTest:
        for test in self.tests:
            if test.id == test_id:
                return test
        raise KeyError(f"No test with id {test_id!r} in this project.")

    def remove_test(self, test_id: str) -> None:
        self.tests = [t for t in self.tests if t.id != test_id]
        self.touch()

    def touch(self) -> None:
        self.modified_at = _now()

    @property
    def analysis_count(self) -> int:
        return sum(len(t.analyses) for t in self.tests)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        def test_dict(t: PumpingTest) -> dict:
            return {
                "id": t.id,
                "name": t.name,
                "pumping_well": t.pumping_well,
                "observation_well": t.observation_well,
                "readings": vars(t.readings) if t.readings else None,
                "setup": vars(t.setup),
                "analyses": [{**{k: v for k, v in vars(a).items() if k != "setup"},
                              "setup": vars(a.setup)} for a in t.analyses],
                # What was run, as against what was saved. Additive: a file
                # written before this field simply has none.
                "runs": [vars(r) for r in t.runs],
            }

        return _sanitise({
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "info": vars(self.info),
            "tests": [test_dict(t) for t in self.tests],
        })

    def to_json(self) -> bytes:
        return json.dumps(self.to_dict(), indent=2, allow_nan=False,
                          ensure_ascii=False).encode("utf-8")

    @property
    def filename(self) -> str:
        stem = "".join(c if c.isalnum() or c in "-_" else "_"
                       for c in (self.info.title or "project").strip())
        return f"{stem.strip('_') or 'project'}{FILE_SUFFIX}"

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            raise ProjectFileError(
                "This is not a HydroAssist project file. Project files are saved "
                f"with the extension {FILE_SUFFIX}.")

        version = data.get("schema_version")
        if not isinstance(version, int):
            raise ProjectFileError("The project file has no valid format version.")
        if version > SCHEMA_VERSION:
            raise ProjectFileError(
                f"This project was saved by a newer version of HydroAssist (format "
                f"{version}; this version reads up to {SCHEMA_VERSION}). Update the "
                "app to open it.")

        try:
            tests = []
            for t in data.get("tests", []):
                readings = Readings(**t["readings"]) if t.get("readings") else None
                analyses = [
                    AnalysisRecord(**{**a, "setup": TestSetup(**a["setup"])})
                    for a in t.get("analyses", [])
                ]
                tests.append(PumpingTest(
                    id=t["id"], name=t["name"],
                    pumping_well=t.get("pumping_well", ""),
                    observation_well=t.get("observation_well", ""),
                    readings=readings,
                    setup=TestSetup(**t.get("setup", {})),
                    analyses=analyses,
                    runs=[RunRecord(**r) for r in (t.get("runs") or [])],
                ))
            return cls(info=ProjectInfo(**data.get("info", {})), tests=tests,
                       created_at=data.get("created_at", _now()),
                       modified_at=data.get("modified_at", _now()))
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectFileError(
                f"The project file is damaged or incomplete ({exc}). It may have "
                "been edited by hand or truncated during download.") from exc

    @classmethod
    def from_json(cls, raw: bytes) -> "Project":
        if len(raw) > MAX_FILE_BYTES:
            raise ProjectFileError(
                f"The file is {len(raw) / 2**20:.0f} MB; project files are limited "
                f"to {MAX_FILE_BYTES // 2**20} MB.")
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProjectFileError(
                "The file could not be read as a HydroAssist project — it is not "
                "valid JSON.") from exc
        return cls.from_dict(data)
