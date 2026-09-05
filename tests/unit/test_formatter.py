"""Unit tests for DataFormatter — unit conversion and cleaning to the standard schema.

The formatter is the only place unit conversion happens: everything downstream
(the calculators) assumes strict SI. A silent conversion error here produces a
plausible-looking T that is wrong by a factor of 60, 3600 or 0.3048, so these
tests check the arithmetic explicitly rather than just the column names.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.csv_inspector import InspectionResult
from src.data.formatter import (
    DataFormatter, FormattedData, TIME_TO_SECONDS, DRAWDOWN_TO_METERS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _inspection(time_col="time_s", dd_cols=None, well_col=None,
                time_unit="seconds", dd_unit="meters") -> InspectionResult:
    return InspectionResult(
        time_col=time_col,
        drawdown_cols=dd_cols if dd_cols is not None else ["drawdown_m"],
        well_id_col=well_col,
        time_unit=time_unit,
        drawdown_unit=dd_unit,
    )


def _fmt(df, insp):
    return DataFormatter().format(df, insp)


# ── Output schema ─────────────────────────────────────────────────────────────

class TestStandardSchema:

    def test_output_columns_are_canonical(self):
        df = pd.DataFrame({"time_s": [1.0, 2.0, 3.0], "drawdown_m": [0.1, 0.2, 0.3]})
        out = _fmt(df, _inspection()).df

        assert list(out.columns) == ["time_s", "drawdown_m"]

    def test_well_id_retained_when_present(self):
        df = pd.DataFrame({
            "time_s": [1.0, 2.0, 3.0],
            "drawdown_m": [0.1, 0.2, 0.3],
            "well": ["MW-1"] * 3,
        })
        out = _fmt(df, _inspection(well_col="well")).df

        assert set(out.columns) == {"time_s", "drawdown_m", "well_id"}
        assert out["well_id"].unique().tolist() == ["MW-1"]

    def test_unrelated_columns_are_dropped(self):
        df = pd.DataFrame({
            "time_s": [1.0, 2.0, 3.0],
            "drawdown_m": [0.1, 0.2, 0.3],
            "operator": ["ab", "cd", "ef"],
            "temperature_c": [20.0, 20.1, 20.2],
        })
        out = _fmt(df, _inspection()).df
        assert list(out.columns) == ["time_s", "drawdown_m"]

    def test_returns_formatted_data_with_shape_bookkeeping(self):
        df = pd.DataFrame({"time_s": [1.0, 2.0, 3.0], "drawdown_m": [0.1, 0.2, 0.3]})
        res = _fmt(df, _inspection())

        assert isinstance(res, FormattedData)
        assert res.original_shape == (3, 2)
        assert res.final_shape == res.df.shape


# ── Unit conversion arithmetic ────────────────────────────────────────────────

class TestTimeConversion:

    def test_minutes_to_seconds(self):
        df = pd.DataFrame({"t": [1.0, 2.0, 10.0], "dd": [0.1, 0.2, 0.3]})
        out = _fmt(df, _inspection("t", ["dd"], time_unit="minutes")).df

        np.testing.assert_allclose(out["time_s"], [60.0, 120.0, 600.0])

    def test_hours_to_seconds(self):
        df = pd.DataFrame({"t": [0.5, 1.0, 2.0], "dd": [0.1, 0.2, 0.3]})
        out = _fmt(df, _inspection("t", ["dd"], time_unit="hours")).df

        np.testing.assert_allclose(out["time_s"], [1800.0, 3600.0, 7200.0])

    def test_seconds_pass_through_unchanged(self):
        df = pd.DataFrame({"t": [1.0, 2.0, 3.0], "dd": [0.1, 0.2, 0.3]})
        out = _fmt(df, _inspection("t", ["dd"], time_unit="seconds")).df

        np.testing.assert_allclose(out["time_s"], [1.0, 2.0, 3.0])

    def test_conversion_is_logged_in_change_list(self):
        df = pd.DataFrame({"t": [1.0, 2.0, 3.0], "dd": [0.1, 0.2, 0.3]})
        res = _fmt(df, _inspection("t", ["dd"], time_unit="minutes"))

        assert any("minutes" in c and "seconds" in c for c in res.conversions_applied)

    def test_unknown_time_unit_assumes_seconds_and_says_so(self):
        """Silent assumptions are the dangerous kind — it must be surfaced."""
        df = pd.DataFrame({"t": [1.0, 2.0, 3.0], "dd": [0.1, 0.2, 0.3]})
        res = _fmt(df, _inspection("t", ["dd"], time_unit="unknown"))

        np.testing.assert_allclose(res.df["time_s"], [1.0, 2.0, 3.0])
        assert any("unknown" in c.lower() for c in res.conversions_applied)

    def test_conversion_table_values(self):
        assert TIME_TO_SECONDS == {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}


class TestDrawdownConversion:

    def test_feet_to_meters(self):
        df = pd.DataFrame({"t": [1.0, 2.0, 3.0], "dd": [1.0, 10.0, 100.0]})
        out = _fmt(df, _inspection("t", ["dd"], dd_unit="feet")).df

        np.testing.assert_allclose(out["drawdown_m"], [0.3048, 3.048, 30.48])

    def test_meters_pass_through_unchanged(self):
        df = pd.DataFrame({"t": [1.0, 2.0, 3.0], "dd": [0.1, 0.2, 0.3]})
        out = _fmt(df, _inspection("t", ["dd"], dd_unit="meters")).df

        np.testing.assert_allclose(out["drawdown_m"], [0.1, 0.2, 0.3])

    def test_unknown_drawdown_unit_assumes_meters_and_says_so(self):
        df = pd.DataFrame({"t": [1.0, 2.0, 3.0], "dd": [0.1, 0.2, 0.3]})
        res = _fmt(df, _inspection("t", ["dd"], dd_unit="unknown"))

        np.testing.assert_allclose(res.df["drawdown_m"], [0.1, 0.2, 0.3])
        assert any("unknown" in c.lower() for c in res.conversions_applied)

    def test_conversion_table_values(self):
        assert DRAWDOWN_TO_METERS == {"meters": 1.0, "feet": 0.3048}

    def test_combined_minutes_and_feet(self):
        """Both axes converted in one pass — the common US field-data case."""
        df = pd.DataFrame({"t": [1.0, 2.0], "dd": [1.0, 2.0]})
        out = _fmt(df, _inspection("t", ["dd"],
                                   time_unit="minutes", dd_unit="feet")).df

        np.testing.assert_allclose(out["time_s"], [60.0, 120.0])
        np.testing.assert_allclose(out["drawdown_m"], [0.3048, 0.6096])


# ── Row cleaning ──────────────────────────────────────────────────────────────

class TestRowCleaning:

    def test_t_zero_row_removed(self):
        """log10(0) is undefined — every method plots against log time."""
        df = pd.DataFrame({
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "drawdown_m": [0.0, 0.1, 0.2, 0.3],
        })
        res = _fmt(df, _inspection())

        assert 0.0 not in res.df["time_s"].values
        assert len(res.df) == 3
        assert any("t=0" in r for r in res.rows_removed)

    def test_duplicate_timestamps_removed_keeping_first(self):
        df = pd.DataFrame({
            "time_s": [1.0, 2.0, 2.0, 3.0],
            "drawdown_m": [0.1, 0.2, 0.99, 0.3],
        })
        res = _fmt(df, _inspection())

        assert len(res.df) == 3
        # The first occurrence (0.2) survives, not the duplicate (0.99)
        assert 0.99 not in res.df["drawdown_m"].values
        assert any("duplicate" in r.lower() for r in res.rows_removed)

    def test_rows_sorted_by_time_ascending(self):
        df = pd.DataFrame({
            "time_s": [10.0, 1.0, 5.0, 2.0],
            "drawdown_m": [0.4, 0.1, 0.3, 0.2],
        })
        out = _fmt(df, _inspection()).df

        assert out["time_s"].is_monotonic_increasing
        np.testing.assert_allclose(out["time_s"], [1.0, 2.0, 5.0, 10.0])
        # drawdown must travel with its own timestamp
        np.testing.assert_allclose(out["drawdown_m"], [0.1, 0.2, 0.3, 0.4])

    def test_non_numeric_rows_dropped(self):
        df = pd.DataFrame({
            "time_s": ["1.0", "2.0", "bad", "4.0"],
            "drawdown_m": ["0.1", "0.2", "0.3", "0.4"],
        })
        res = _fmt(df, _inspection())

        assert len(res.df) == 3
        assert any("non-numeric" in r.lower() for r in res.rows_removed)

    def test_index_is_reset_after_cleaning(self):
        """Calculators index with .to_numpy(); a stale index masks alignment bugs."""
        df = pd.DataFrame({
            "time_s": [0.0, 5.0, 1.0, 5.0],
            "drawdown_m": [0.0, 0.3, 0.1, 0.9],
        })
        out = _fmt(df, _inspection()).df

        assert out.index.tolist() == list(range(len(out)))


# ── Wide → long ───────────────────────────────────────────────────────────────

class TestWideToLong:

    def test_multi_well_wide_format_melted(self):
        df = pd.DataFrame({
            "time_s": [1.0, 2.0, 3.0],
            "mw1_dd_m": [0.1, 0.2, 0.3],
            "mw2_dd_m": [0.05, 0.1, 0.15],
        })
        res = _fmt(df, _inspection(dd_cols=["mw1_dd_m", "mw2_dd_m"]))

        assert "well_id" in res.df.columns
        assert len(res.df) == 6
        assert set(res.df["well_id"].unique()) == {"mw1_dd_m", "mw2_dd_m"}
        assert any("melt" in c.lower() for c in res.conversions_applied)

    def test_melted_values_stay_with_their_well(self):
        df = pd.DataFrame({
            "time_s": [1.0, 2.0],
            "mw1_dd_m": [0.1, 0.2],
            "mw2_dd_m": [0.9, 0.8],
        })
        out = _fmt(df, _inspection(dd_cols=["mw1_dd_m", "mw2_dd_m"])).df

        mw1 = out[out["well_id"] == "mw1_dd_m"].sort_values("time_s")
        np.testing.assert_allclose(mw1["drawdown_m"], [0.1, 0.2])

    def test_units_converted_across_all_melted_wells(self):
        df = pd.DataFrame({
            "t": [1.0, 2.0],
            "mw1_ft": [1.0, 2.0],
            "mw2_ft": [10.0, 20.0],
        })
        out = _fmt(df, _inspection("t", ["mw1_ft", "mw2_ft"],
                                   time_unit="minutes", dd_unit="feet")).df

        np.testing.assert_allclose(sorted(out["time_s"].unique()), [60.0, 120.0])
        np.testing.assert_allclose(
            sorted(out["drawdown_m"]), sorted([0.3048, 0.6096, 3.048, 6.096])
        )

    def test_duplicates_deduplicated_per_well_not_globally(self):
        """Two wells legitimately share timestamps — that is not duplication."""
        df = pd.DataFrame({
            "time_s": [1.0, 2.0],
            "mw1_m": [0.1, 0.2],
            "mw2_m": [0.5, 0.6],
        })
        out = _fmt(df, _inspection(dd_cols=["mw1_m", "mw2_m"])).df

        assert len(out) == 4, "shared timestamps across wells must be preserved"


# ── Round trip against the real sample files ──────────────────────────────────

class TestBundledSamples:

    @pytest.mark.parametrize("path", [
        "tests/theis_cooper_jacob_sample.csv",
        "tests/papadopulos_cooper_sample.csv",
    ])
    def test_sample_survives_inspect_then_format(self, path):
        from src.data.csv_inspector import CSVInspector

        raw = pd.read_csv(path)
        insp = CSVInspector().inspect(raw)
        res = DataFormatter().format(raw, insp)

        assert list(res.df.columns) == ["time_s", "drawdown_m"]
        assert len(res.df) == len(raw), "no rows should be dropped from clean data"
        assert (res.df["time_s"] > 0).all()
        assert res.df["time_s"].is_monotonic_increasing
        # Values must be untouched — the samples are already SI
        np.testing.assert_allclose(
            res.df["time_s"].to_numpy(), raw["time_s"].to_numpy()
        )
