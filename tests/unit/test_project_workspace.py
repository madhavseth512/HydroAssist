"""Tests for the workspace ↔ project glue.

These are the rules that decide whether a reopened project looks the way it was
left: the Step 2 form must come back with the same Q in the same unit, values
must never land outside a widget's bounds (Streamlit raises if they do), and a
saved analysis must record the setup its fit actually used rather than whatever
the form shows at the moment of saving.
"""

import numpy as np
import pandas as pd
import pytest

from src.calculator.base import CalculationInput
from src.calculator.methods.hantush_jacob import (
    HantushJacobCalculator, hantush_jacob_drawdown,
)
from src.calculator.theis import TheisCalculator
from src.project.model import Readings, TestSetup
from src.project.workspace import (
    BOUNDS, DEFAULTS, KEY_CLOCK, KEY_Q, KEY_Q_UNIT, KEY_R, KEY_RC, KEY_RW,
    KEY_TEST_TYPE, KEY_TP, Q_UNITS,
    comparison_frame, inputs_from_setup, new_project, project_summary,
    q_to_si, setup_from_inputs, setup_from_run, stale_count,
)


class TestUnits:

    @pytest.mark.parametrize("value,unit,expected", [
        (86400.0, "m³/day", 1.0),
        (3600.0, "m³/hour", 1.0),
        (1.0, "m³/s", 1.0),
        (1000.0, "L/s", 1.0),
    ])
    def test_q_converts_to_si(self, value, unit, expected):
        assert q_to_si(value, unit) == pytest.approx(expected)

    def test_unknown_unit_is_rejected_with_the_valid_options(self):
        with pytest.raises(ValueError, match="m³/day"):
            q_to_si(1.0, "gallons/min")

    def test_the_unit_list_matches_the_conversions(self):
        assert Q_UNITS == ["m³/day", "m³/hour", "m³/s", "L/s"]


class TestFormRoundTrip:

    def test_a_setup_reproduces_the_form_that_made_it(self):
        form = {KEY_Q: 12.5, KEY_Q_UNIT: "L/s", KEY_R: 35.0,
                KEY_RW: 0.3, KEY_RC: 0.2, KEY_TEST_TYPE: "recovery",
                KEY_TP: 12.0, KEY_CLOCK: "since_start"}
        assert inputs_from_setup(setup_from_inputs(form)) == form

    def test_the_entered_unit_is_kept_not_converted(self):
        """Reopening must show 12.5 L/s, not 1080 m³/day."""
        setup = setup_from_inputs({KEY_Q: 12.5, KEY_Q_UNIT: "L/s"})
        values = inputs_from_setup(setup)

        assert values[KEY_Q] == pytest.approx(12.5)
        assert values[KEY_Q_UNIT] == "L/s"

    def test_missing_keys_fall_back_to_defaults(self):
        """Streamlit drops the state of widgets not rendered in a run."""
        setup = setup_from_inputs({})
        assert setup.Q_entered == DEFAULTS[KEY_Q]
        assert setup.r_w_m == DEFAULTS[KEY_RW]

    def test_si_rate_is_computed_from_the_entered_value(self):
        setup = setup_from_inputs({KEY_Q: 500.0, KEY_Q_UNIT: "m³/day"})
        assert setup.Q_m3s == pytest.approx(500 / 86400)

    def test_an_unknown_stored_unit_is_re_expressed_from_the_si_rate(self):
        """A hand-edited or future file with an unfamiliar unit must still show
        the right rate rather than the number in the wrong unit."""
        setup = TestSetup(Q_m3s=500 / 86400, Q_entered=132.0, Q_unit="gpm")
        values = inputs_from_setup(setup)

        assert values[KEY_Q_UNIT] == "m³/day"
        assert values[KEY_Q] == pytest.approx(500.0)

    def test_values_are_clamped_to_widget_bounds(self):
        """Out-of-range values would make Streamlit raise on render."""
        setup = TestSetup(Q_m3s=None, Q_entered=1e9, Q_unit="m³/day",
                          r_m=1e6, r_w_m=0.0001, r_c_m=99.0)
        values = inputs_from_setup(setup)

        for key in (KEY_Q, KEY_R, KEY_RW, KEY_RC):
            lo, hi = BOUNDS[key]
            assert lo <= values[key] <= hi, key


class TestSetupFromTheRun:

    def test_records_what_the_fit_used(self):
        run = {"Q": 300 / 86400, "Q_entered": 300.0, "Q_unit": "m³/day",
               "r": 40.0, "r_w": 0.3, "r_c": 0.2}
        setup = setup_from_run(run)

        assert setup.Q_m3s == pytest.approx(300 / 86400)
        assert setup.Q_entered == 300.0
        assert (setup.r_m, setup.r_w_m, setup.r_c_m) == (40.0, 0.3, 0.2)

    def test_is_independent_of_the_form(self):
        """Editing Q after running must not change what the saved analysis says."""
        run = {"Q": 300 / 86400, "Q_entered": 300.0, "Q_unit": "m³/day", "r": 40.0}
        later_form = {KEY_Q: 999.0, KEY_Q_UNIT: "m³/day", KEY_R: 1.0}

        assert setup_from_run(run).Q_entered == 300.0
        assert setup_from_inputs(later_form).Q_entered == 999.0


class TestProjectHelpers:

    def test_a_new_project_has_one_empty_test(self):
        project = new_project()
        assert len(project.tests) == 1
        assert project.tests[0].readings is None

    @pytest.mark.parametrize("n_tests,expected", [
        (1, "1 test · 0 saved analyses"),
        (2, "2 tests · 0 saved analyses"),
    ])
    def test_summary_pluralises(self, n_tests, expected):
        project = new_project()
        for _ in range(n_tests - 1):
            project.add_test()
        assert project_summary(project) == expected


def _leaky():
    t = np.logspace(1, 6, 70)
    s = hantush_jacob_drawdown(t, 2.5e-3, 5e-5, 300.0, 500 / 86400, 50.0)
    s = s + np.random.default_rng(2).normal(0, 0.002, t.size)
    return pd.DataFrame({"time_s": t, "drawdown_m": s})


def _test_with_two_analyses(df):
    project = new_project()
    test = project.tests[0]
    test.set_readings(Readings.from_frame(df))
    for calc, label in ((TheisCalculator, "Theis (1935)"),
                        (HantushJacobCalculator, "Hantush-Jacob (1955)")):
        test.add_analysis(
            calc().calculate(CalculationInput(df=df, Q=500 / 86400, r=50.0)), label)
    return test


class TestComparisonFrame:

    def test_empty_test_gives_an_empty_frame(self):
        assert comparison_frame(new_project().tests[0]).empty

    def test_best_model_is_ranked_first(self):
        frame = comparison_frame(_test_with_two_analyses(_leaky()))

        assert list(frame["Method"]) == ["Hantush-Jacob (1955)", "Theis (1935)"]
        assert list(frame["Rank"]) == [1, 2]

    def test_evidence_uses_the_conventional_thresholds(self):
        frame = comparison_frame(_test_with_two_analyses(_leaky()))

        assert frame.iloc[0]["Evidence"] == "best"
        assert frame.iloc[1]["Evidence"] == "essentially no support"

    def test_columns_explain_the_ranking(self):
        frame = comparison_frame(_test_with_two_analyses(_leaky()))
        for column in ("AICc", "ΔAICc", "Weight", "R²", "Parameters"):
            assert column in frame.columns

    def test_stale_analyses_are_counted_and_excluded(self):
        test = _test_with_two_analyses(_leaky())
        test.set_readings(Readings.from_frame(_leaky().iloc[::2]))

        assert stale_count(test) == 2
        assert comparison_frame(test).empty
