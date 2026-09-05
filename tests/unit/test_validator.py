"""Unit tests for DataValidator — plausibility checks and method suitability.

The validator is advisory: it never modifies data. Its job is to (a) block
analysis that cannot succeed, (b) warn about data that will fit badly, and
(c) narrow the method list the UI offers. These tests pin down which findings
are blocking (errors) versus advisory (warnings/info), because the UI gates the
Run button on `is_usable`.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.validator import (
    DataValidator, ValidationResult, MIN_DATA_POINTS, MIN_LOG_CYCLES,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _good_df(n=30, t_min=10.0, t_max=10000.0):
    """A dataset that passes every check: enough points, monotonic, >1 log cycle."""
    t = np.logspace(np.log10(t_min), np.log10(t_max), n)
    s = 0.15 * np.log10(t / (t_min / 10.0))
    return pd.DataFrame({"time_s": t, "drawdown_m": s})


def _validate(df, **kw):
    return DataValidator().validate(df, **kw)


# ── Blocking errors ───────────────────────────────────────────────────────────

class TestBlockingErrors:

    def test_clean_dataset_is_usable(self):
        res = _validate(_good_df())

        assert isinstance(res, ValidationResult)
        assert res.is_usable is True
        assert res.errors == []

    def test_too_few_points_blocks_analysis(self):
        res = _validate(_good_df(n=MIN_DATA_POINTS - 1))

        assert res.is_usable is False
        assert any("data points" in e for e in res.errors)

    def test_exactly_minimum_points_is_allowed(self):
        """Boundary: the minimum itself must pass, not fail."""
        res = _validate(_good_df(n=MIN_DATA_POINTS))
        assert res.is_usable is True

    def test_error_message_names_the_dataset(self):
        res = _validate(_good_df(n=4))
        assert any("Dataset" in e for e in res.errors)


# ── Non-blocking warnings ─────────────────────────────────────────────────────

class TestWarnings:

    def test_non_monotonic_time_warns_but_does_not_block(self):
        df = _good_df()
        df.loc[5, "time_s"] = df.loc[2, "time_s"]  # step backwards
        res = _validate(df)

        assert res.is_usable is True
        assert any("monotonic" in w.lower() for w in res.warnings)

    def test_negative_drawdown_warns(self):
        df = _good_df()
        df.loc[0, "drawdown_m"] = -0.05
        res = _validate(df)

        assert any("negative" in w.lower() for w in res.warnings)

    def test_short_log_cycle_span_warns(self):
        """Under one log cycle, T and S trade off and the fit is unstable."""
        res = _validate(_good_df(n=20, t_min=100.0, t_max=300.0))

        assert any("log cycle" in w.lower() for w in res.warnings)

    def test_adequate_log_span_does_not_warn(self):
        res = _validate(_good_df(n=20, t_min=10.0, t_max=10000.0))
        assert not any("log cycle" in w.lower() for w in res.warnings)

    def test_log_cycle_threshold_constant(self):
        assert MIN_LOG_CYCLES == 1.0

    def test_decreasing_early_drawdown_is_flagged(self):
        """Recovery data pasted in as pumping data — drawdown falls, not rises."""
        df = _good_df()
        df["drawdown_m"] = df["drawdown_m"].values[::-1]
        res = _validate(df)

        assert any("recovery" in i.lower() for i in res.info)


# ── Multi-well handling ───────────────────────────────────────────────────────

class TestMultiWell:

    def _two_wells(self, n_a=30, n_b=30):
        a = _good_df(n=n_a); a["well_id"] = "MW-1"
        b = _good_df(n=n_b); b["well_id"] = "MW-2"
        return pd.concat([a, b], ignore_index=True)

    def test_each_well_checked_independently(self):
        res = _validate(self._two_wells(n_a=30, n_b=4))

        assert res.is_usable is False
        assert any("MW-2" in e for e in res.errors)
        assert not any("MW-1" in e for e in res.errors)

    def test_findings_are_labelled_per_well(self):
        df = self._two_wells()
        df.loc[df["well_id"] == "MW-2", "drawdown_m"] = -0.1
        res = _validate(df)

        assert any("MW-2" in w for w in res.warnings)

    def test_both_wells_good_is_usable(self):
        res = _validate(self._two_wells())
        assert res.is_usable is True


# ── Method suitability ────────────────────────────────────────────────────────

class TestMethodSuitability:

    def test_theis_always_offered_for_usable_data(self):
        res = _validate(_good_df())
        assert "Theis (1935)" in res.suitable_methods

    def test_papadopulos_cooper_always_offered(self):
        """PC needs extra geometry, so it is offered with an explanatory note."""
        res = _validate(_good_df())

        assert "Papadopulos-Cooper (1967)" in res.suitable_methods
        assert any("Papadopulos" in i for i in res.info)

    def test_cooper_jacob_offered_for_long_tests(self):
        res = _validate(_good_df(t_min=10.0, t_max=20000.0),
                        well_radius_m=50.0, pumping_rate_m3d=500.0)
        assert "Cooper-Jacob (1946)" in res.suitable_methods

    def test_cooper_jacob_withheld_for_very_short_tests(self):
        """Under 30 min the late-time u < 0.02 requirement is unlikely to hold."""
        res = _validate(_good_df(t_min=1.0, t_max=600.0),
                        well_radius_m=50.0, pumping_rate_m3d=500.0)

        assert "Cooper-Jacob (1946)" not in res.suitable_methods
        assert any("Cooper-Jacob" in i for i in res.info)

    def test_cooper_jacob_offered_when_geometry_unknown(self):
        """Without r and Q the check cannot run — be permissive, not restrictive."""
        res = _validate(_good_df(t_min=1.0, t_max=600.0))
        assert "Cooper-Jacob (1946)" in res.suitable_methods

    def test_large_casing_radius_recommends_papadopulos_cooper(self):
        """Theis assumption (5) is an infinitesimal-diameter well."""
        res = _validate(_good_df(), well_casing_radius_m=0.30)

        assert any("Papadopulos-Cooper" in i and "large-diameter" in i
                   for i in res.info)

    def test_small_casing_radius_does_not_trigger_recommendation(self):
        res = _validate(_good_df(), well_casing_radius_m=0.05)

        assert not any("infinitesimal" in i for i in res.info)

    def test_suitable_methods_never_contains_unknown_names(self):
        res = _validate(_good_df(), well_radius_m=50.0, pumping_rate_m3d=500.0)
        assert set(res.suitable_methods) <= set(DataValidator.ALL_METHODS)


# ── The validator must not mutate its input ───────────────────────────────────

class TestReadOnly:

    def test_input_dataframe_is_unchanged(self):
        df = _good_df()
        before = df.copy(deep=True)
        _validate(df)

        pd.testing.assert_frame_equal(df, before)


# ── Integration with the real pipeline ────────────────────────────────────────

class TestPipelineIntegration:

    @pytest.mark.parametrize("path", [
        "tests/theis_cooper_jacob_sample.csv",
        "tests/papadopulos_cooper_sample.csv",
    ])
    def test_bundled_samples_validate_clean(self, path):
        from src.data.csv_inspector import CSVInspector
        from src.data.formatter import DataFormatter

        raw = pd.read_csv(path)
        insp = CSVInspector().inspect(raw)
        formatted = DataFormatter().format(raw, insp).df
        res = DataValidator().validate(formatted)

        assert res.is_usable is True, f"{path}: {res.errors}"
        assert res.errors == []
        assert "Theis (1935)" in res.suitable_methods
