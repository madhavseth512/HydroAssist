"""Unit tests for CSVInspector — column detection, unit inference, anomaly scans.

The inspector is the first thing that touches user data, so these tests focus on
the shapes of real field CSVs: unit suffixes in several spellings, parenthesised
units, wide multi-well layouts, and dirty data (missing values, unsorted rows,
t=0, recovery-phase negatives).

Layer 3 (LLM fallback) is exercised with a stub LLM — no network access.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.data.csv_inspector import CSVInspector, InspectionResult


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def _df(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


def _clean_test_data(n=20):
    """A well-formed single-well pumping test."""
    t = np.logspace(1, 4, n)
    return t, 0.05 * np.log10(t) * 3


class _StubLLM:
    """Minimal stand-in for a LangChain chat model.

    Records the prompt it was handed and returns a canned response object with
    a `.content` attribute, matching what CSVInspector consumes.
    """

    def __init__(self, payload, raise_on_call=False, wrap_markdown=False):
        self.payload = payload
        self.raise_on_call = raise_on_call
        self.wrap_markdown = wrap_markdown
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        if self.raise_on_call:
            raise RuntimeError("LLM unavailable")
        body = json.dumps(self.payload)
        if self.wrap_markdown:
            body = f"```json\n{body}\n```"
        return type("Resp", (), {"content": body})()


# ── Layer 1: regex column detection ───────────────────────────────────────────

class TestColumnDetection:

    def test_canonical_schema_is_detected(self):
        """time_s / drawdown_m is the schema this project emits and ships."""
        t, s = _clean_test_data()
        res = CSVInspector().inspect(_df(time_s=t, drawdown_m=s))

        assert res.time_col == "time_s"
        assert res.drawdown_cols == ["drawdown_m"]
        assert res.well_id_col is None

    @pytest.mark.parametrize("time_col,expected", [
        ("time_s", "seconds"),
        ("time_sec", "seconds"),
        ("elapsed_seconds", "seconds"),
        ("Time (s)", "seconds"),
        ("time_min", "minutes"),
        ("time_mins", "minutes"),
        ("elapsed_minutes", "minutes"),
        ("Time (min)", "minutes"),
        ("time_hr", "hours"),
        ("time_hrs", "hours"),
        ("elapsed_hours", "hours"),
        ("Elapsed Time (hr)", "hours"),
    ])
    def test_time_unit_inference(self, time_col, expected):
        t, s = _clean_test_data()
        res = CSVInspector().inspect(_df(**{time_col: t, "drawdown_m": s}))
        assert res.time_unit == expected

    @pytest.mark.parametrize("dd_col,expected", [
        ("drawdown_m", "meters"),
        ("drawdown_meters", "meters"),
        ("Drawdown (m)", "meters"),
        ("dd_metres", "meters"),
        ("drawdown_ft", "feet"),
        ("drawdown_feet", "feet"),
        ("Drawdown (ft)", "feet"),
    ])
    def test_drawdown_unit_inference(self, dd_col, expected):
        t, s = _clean_test_data()
        res = CSVInspector().inspect(_df(**{"time_s": t, dd_col: s}))
        assert res.drawdown_unit == expected

    def test_unitless_names_lower_confidence(self):
        """No unit token in the name → flagged for user confirmation."""
        t, s = _clean_test_data()
        res = CSVInspector().inspect(_df(elapsed=t, water_level=s))

        assert res.time_unit == "unknown"
        assert res.drawdown_unit == "unknown"
        assert res.confidence < CSVInspector.CONFIDENCE_THRESHOLD
        assert res.needs_user_confirmation is True

    def test_unit_token_must_be_a_whole_token(self):
        """'m' inside a word is not a metre; 's' inside a word is not a second."""
        t, s = _clean_test_data()
        # "monitoring" contains 'm'; "elapsed" contains 's'
        res = CSVInspector().inspect(_df(elapsed=t, monitoring_dd=s))
        assert res.time_unit == "unknown"
        assert res.drawdown_unit == "unknown"

    def test_well_id_column_detected(self):
        t, s = _clean_test_data(10)
        res = CSVInspector().inspect(
            _df(time_s=t, drawdown_m=s, well_id=["MW-1"] * 10)
        )
        assert res.well_id_col == "well_id"

    def test_well_id_not_confused_with_time_or_drawdown(self):
        """The well column must not be one already claimed by time/drawdown."""
        t, s = _clean_test_data(10)
        res = CSVInspector().inspect(
            _df(time_s=t, drawdown_m=s, borehole=["BH-2"] * 10)
        )
        assert res.well_id_col == "borehole"
        assert res.time_col == "time_s"
        assert res.drawdown_cols == ["drawdown_m"]

    def test_wide_multi_well_returns_all_drawdown_columns(self):
        t, s = _clean_test_data(15)
        res = CSVInspector().inspect(
            _df(time_min=t, drawdown_mw1_m=s, drawdown_mw2_m=s * 0.6)
        )
        assert set(res.drawdown_cols) == {"drawdown_mw1_m", "drawdown_mw2_m"}


# ── Layer 2: statistical fallback ─────────────────────────────────────────────

class TestStatisticalDetection:

    def test_monotonic_column_detected_as_time_when_name_is_opaque(self):
        """No regex hit on names → fall back to shape of the data."""
        t = np.linspace(0, 1000, 30)
        s = np.linspace(0, 2.5, 30)
        res = CSVInspector().inspect(_df(colA=t, colB=s))

        assert res.time_col == "colA"
        assert res.confidence < CSVInspector.CONFIDENCE_THRESHOLD

    def test_drawdown_fallback_picks_plausible_range(self):
        t = np.linspace(0, 1000, 30)
        s = np.linspace(0, 2.5, 30)
        res = CSVInspector().inspect(_df(colA=t, colB=s))
        assert res.drawdown_cols  # something was chosen

    def test_no_numeric_columns_yields_zero_confidence(self):
        res = CSVInspector().inspect(
            _df(label=["a", "b", "c"], note=["x", "y", "z"])
        )
        assert res.confidence == 0.0
        assert res.needs_user_confirmation is True
        assert any("time" in i.lower() for i in res.issues)


# ── Layer 3: LLM fallback ─────────────────────────────────────────────────────

class TestLLMFallback:

    def _ambiguous(self):
        t = np.linspace(1, 500, 20)
        return _df(elapsed=t, water_level=np.linspace(0.1, 3.0, 20))

    def test_llm_not_called_when_confidence_is_high(self):
        """A confident regex match must not spend an API call."""
        t, s = _clean_test_data()
        llm = _StubLLM({})
        CSVInspector(llm=llm).inspect(_df(time_min=t, drawdown_ft=s))
        assert llm.calls == []

    def test_llm_called_and_applied_when_confidence_is_low(self):
        llm = _StubLLM({
            "time_column": "elapsed",
            "time_unit": "minutes",
            "drawdown_columns": ["water_level"],
            "drawdown_unit": "feet",
            "explanation": "Elapsed is minutes since pump start.",
        })
        res = CSVInspector(llm=llm).inspect(self._ambiguous())

        assert len(llm.calls) == 1
        assert res.time_unit == "minutes"
        assert res.drawdown_unit == "feet"
        assert res.confidence >= 0.85
        assert "minutes" in res.llm_explanation

    def test_llm_response_wrapped_in_markdown_is_parsed(self):
        """Gemini frequently wraps JSON in ```json fences."""
        llm = _StubLLM({
            "time_column": "elapsed",
            "time_unit": "hours",
            "drawdown_columns": ["water_level"],
            "drawdown_unit": "meters",
            "explanation": "ok",
        }, wrap_markdown=True)
        res = CSVInspector(llm=llm).inspect(self._ambiguous())

        assert res.time_unit == "hours"
        assert res.drawdown_unit == "meters"

    def test_llm_failure_degrades_gracefully(self):
        """An LLM outage must not fail the upload — fall back to layers 1-2."""
        llm = _StubLLM({}, raise_on_call=True)
        res = CSVInspector(llm=llm).inspect(self._ambiguous())

        assert isinstance(res, InspectionResult)
        assert res.time_col == "elapsed"
        assert res.needs_user_confirmation is True

    def test_no_llm_configured_is_not_an_error(self):
        res = CSVInspector(llm=None).inspect(self._ambiguous())
        assert isinstance(res, InspectionResult)
        assert res.llm_explanation == ""


# ── Anomaly scanning ──────────────────────────────────────────────────────────

class TestAnomalyScans:

    def test_missing_time_values_reported(self):
        res = CSVInspector().inspect(_df(
            time_s=[1.0, 2.0, np.nan, 4.0],
            drawdown_m=[0.1, 0.2, 0.3, 0.4],
        ))
        assert any("missing" in i.lower() for i in res.issues)

    def test_non_monotonic_time_reported(self):
        res = CSVInspector().inspect(_df(
            time_s=[1.0, 5.0, 3.0, 9.0],
            drawdown_m=[0.1, 0.2, 0.3, 0.4],
        ))
        assert any("monotonic" in i.lower() for i in res.issues)

    def test_t_zero_row_reported(self):
        res = CSVInspector().inspect(_df(
            time_s=[0.0, 1.0, 2.0, 3.0],
            drawdown_m=[0.0, 0.1, 0.2, 0.3],
        ))
        assert any("t=0" in i for i in res.issues)

    def test_negative_drawdown_reported(self):
        """Negative drawdown usually means recovery data was left in."""
        res = CSVInspector().inspect(_df(
            time_s=[1.0, 2.0, 3.0, 4.0],
            drawdown_m=[0.4, 0.2, -0.1, -0.3],
        ))
        assert any("negative" in i.lower() for i in res.issues)

    def test_clean_data_produces_no_issues(self):
        t, s = _clean_test_data()
        res = CSVInspector().inspect(_df(time_s=t, drawdown_m=s))
        assert res.issues == []
        assert res.needs_user_confirmation is False


# ── Real bundled sample files ─────────────────────────────────────────────────

class TestBundledSamples:
    """The sample CSVs shipped with the repo must inspect cleanly."""

    @pytest.mark.parametrize("path", [
        "tests/theis_cooper_jacob_sample.csv",
        "tests/papadopulos_cooper_sample.csv",
    ])
    def test_sample_file_inspects_with_full_confidence(self, path):
        res = CSVInspector().inspect(pd.read_csv(path))

        assert res.time_col == "time_s"
        assert res.drawdown_cols == ["drawdown_m"]
        assert res.time_unit == "seconds"
        assert res.drawdown_unit == "meters"
        assert res.needs_user_confirmation is False, (
            "Bundled samples must not require manual unit confirmation"
        )
