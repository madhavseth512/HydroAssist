"""Tests for the Hantush-Jacob (1955) leaky confined aquifer calculator.

This is the first method built entirely on the shared machinery, so these tests
serve two purposes: they validate the method, and they are the evidence for
whether the architecture refactor actually made new methods cheap.

The acceptance gates are the same three every subsequent method must pass:

1. **Analytical limits.** Two exist for Hantush-Jacob and both are exact, so
   neither depends on a table transcribed from a paper:
     * B -> infinity reproduces Theis, computable in closed form with expn
     * t -> infinity reaches the steady state s_max = Q/(2*pi*T)*K0(r/B),
       paper eq. 9
   Passing both means the characteristic root and the shared kernel are right.

2. **Round-trip recovery** of known parameters from synthetic noisy data.

3. **Physics invariants** — positivity, monotonicity, linearity in Q, the right
   direction of response to each parameter.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expn, kv

from src.calculator.base import CalculationInput
from src.calculator.methods.hantush_jacob import (
    B_MAX, B_MIN, NEGLIGIBLE_R_OVER_B, HantushJacobCalculator,
    hantush_jacob_drawdown, steady_state_drawdown,
)

Q, R = 500 / 86400, 50.0
T_TRUE, S_TRUE, B_TRUE = 2.5e-3, 5e-5, 300.0


def _theis(t, Q, T, S, r):
    u = (r ** 2 * S) / (4.0 * T * np.asarray(t, dtype=float))
    return (Q / (4.0 * np.pi * T)) * expn(1, np.maximum(u, 1e-12))


def _synthetic(noise=0.003, n=60, seed=42, B=B_TRUE):
    t = np.logspace(1, 5, n)
    s = hantush_jacob_drawdown(t, T_TRUE, S_TRUE, B, Q, R)
    if noise:
        s = s + np.random.default_rng(seed).normal(0, noise, t.size)
    return pd.DataFrame({"time_s": t, "drawdown_m": s})


@pytest.fixture(scope="module")
def fit():
    return HantushJacobCalculator().calculate(
        CalculationInput(df=_synthetic(), Q=Q, r=R))


# ── Gate 1: analytical limits ─────────────────────────────────────────────────

class TestAnalyticalLimits:
    """The acceptance gate. Both limits are exact, not transcribed."""

    def test_no_leakage_reproduces_theis(self):
        """B -> infinity removes leakage entirely."""
        t = np.logspace(1, 5, 30)
        leaky = hantush_jacob_drawdown(t, T_TRUE, S_TRUE, 1e9, Q, R)
        theis = _theis(t, Q, T_TRUE, S_TRUE, R)

        assert np.max(np.abs(leaky - theis) / theis) < 5e-3

    def test_late_time_reaches_the_steady_state_of_equation_9(self):
        """s_max = Q/(2*pi*T)*K0(r/B). Theis has no such limit — drawdown there
        grows without bound — so this is the signature of a leaky aquifer."""
        t = np.logspace(6, 9, 20)
        s = hantush_jacob_drawdown(t, T_TRUE, S_TRUE, B_TRUE, Q, R)
        expected = steady_state_drawdown(Q, T_TRUE, R, B_TRUE)

        assert s[-1] == pytest.approx(expected, rel=1e-3)

    def test_steady_state_formula_matches_the_paper(self):
        expected = Q / (2.0 * np.pi * T_TRUE) * kv(0, R / B_TRUE)
        assert steady_state_drawdown(Q, T_TRUE, R, B_TRUE) == pytest.approx(expected)

    def test_drawdown_stops_growing_at_late_time(self):
        t = np.logspace(6, 9, 20)
        s = hantush_jacob_drawdown(t, T_TRUE, S_TRUE, B_TRUE, Q, R)

        assert abs(s[-1] - s[-5]) / s[-1] < 1e-3

    def test_theis_by_contrast_keeps_growing(self):
        """Contrast test, so the previous assertion cannot pass vacuously."""
        t = np.logspace(6, 9, 20)
        s = _theis(t, Q, T_TRUE, S_TRUE, R)

        assert abs(s[-1] - s[-5]) / s[-1] > 0.05

    def test_leakage_always_reduces_drawdown(self):
        t = np.logspace(2, 6, 25)
        leaky = hantush_jacob_drawdown(t, T_TRUE, S_TRUE, B_TRUE, Q, R)
        theis = _theis(t, Q, T_TRUE, S_TRUE, R)

        assert np.all(leaky <= theis * (1 + 1e-9))


# ── Gate 2: round-trip recovery ───────────────────────────────────────────────

class TestParameterRecovery:

    def test_transmissivity_within_two_percent(self, fit):
        assert fit.success, fit.error_message
        assert fit.value("T") == pytest.approx(T_TRUE, rel=0.02)

    def test_storativity_within_ten_percent(self, fit):
        assert fit.value("S") == pytest.approx(S_TRUE, rel=0.10)

    def test_leakage_factor_within_ten_percent(self, fit):
        assert fit.value("B") == pytest.approx(B_TRUE, rel=0.10)

    def test_fit_quality_is_high(self, fit):
        assert fit.r_squared > 0.999

    def test_noise_free_data_recovers_almost_exactly(self):
        result = HantushJacobCalculator().calculate(
            CalculationInput(df=_synthetic(noise=0.0), Q=Q, r=R))

        assert result.value("T") == pytest.approx(T_TRUE, rel=1e-3)
        assert result.value("B") == pytest.approx(B_TRUE, rel=1e-3)

    def test_recovers_across_a_range_of_leakage(self):
        for B in (100.0, 500.0, 2000.0):
            result = HantushJacobCalculator().calculate(
                CalculationInput(df=_synthetic(noise=0.001, B=B), Q=Q, r=R))
            assert result.value("B") == pytest.approx(B, rel=0.15), f"B={B}"


# ── Gate 3: physics invariants ────────────────────────────────────────────────

class TestPhysicsInvariants:

    def test_drawdown_is_positive_and_never_decreases(self):
        """Not strictly increasing, unlike Theis. Once leakage balances pumping
        the curve is flat, and there it is level to within the inverter's ~1e-6
        absolute noise floor — so the invariant is non-decreasing, not increasing.
        Asserting strict monotonicity here would be asserting the wrong physics.
        """
        t = np.logspace(1, 5, 30)
        s = hantush_jacob_drawdown(t, T_TRUE, S_TRUE, B_TRUE, Q, R)

        assert np.all(s > 0)
        assert np.all(np.diff(s) > -1e-5)

    def test_drawdown_increases_strictly_before_the_plateau(self):
        """Early and mid time, before leakage balances pumping, it does climb."""
        t = np.logspace(1, 3.5, 25)
        s = hantush_jacob_drawdown(t, T_TRUE, S_TRUE, B_TRUE, Q, R)

        assert np.all(np.diff(s) > 0)

    def test_drawdown_is_linear_in_discharge(self):
        t = np.logspace(2, 4, 10)
        single = hantush_jacob_drawdown(t, T_TRUE, S_TRUE, B_TRUE, Q, R)
        double = hantush_jacob_drawdown(t, T_TRUE, S_TRUE, B_TRUE, 2 * Q, R)

        np.testing.assert_allclose(double, 2 * single, rtol=1e-9)

    def test_drawdown_decreases_with_distance(self):
        t = np.array([1e4])
        near = hantush_jacob_drawdown(t, T_TRUE, S_TRUE, B_TRUE, Q, 10.0)
        far = hantush_jacob_drawdown(t, T_TRUE, S_TRUE, B_TRUE, Q, 200.0)

        assert near[0] > far[0]

    def test_higher_transmissivity_means_less_drawdown(self):
        t = np.array([1e4])
        low = hantush_jacob_drawdown(t, 1e-3, S_TRUE, B_TRUE, Q, R)
        high = hantush_jacob_drawdown(t, 1e-2, S_TRUE, B_TRUE, Q, R)

        assert high[0] < low[0]

    def test_stronger_leakage_means_less_drawdown(self):
        t = np.array([1e5])
        weak = hantush_jacob_drawdown(t, T_TRUE, S_TRUE, 2000.0, Q, R)
        strong = hantush_jacob_drawdown(t, T_TRUE, S_TRUE, 100.0, Q, R)

        assert strong[0] < weak[0]

    def test_remains_finite_over_ten_log_cycles(self):
        t = np.logspace(-2, 8, 60)
        s = hantush_jacob_drawdown(t, T_TRUE, S_TRUE, B_TRUE, Q, R)

        assert np.all(np.isfinite(s))


# ── Result shape and interpretation ───────────────────────────────────────────

class TestResultShape:

    def test_reports_three_fitted_parameters_plus_one_derived(self, fit):
        assert fit.parameter_names == ["T", "S", "B", "r_over_B"]

    def test_r_over_B_is_derived_not_fitted(self, fit):
        assert fit.get("r_over_B").derived is True
        assert fit.get("r_over_B").estimated is False

    def test_r_over_B_is_consistent_with_B(self, fit):
        assert fit.value("r_over_B") == pytest.approx(R / fit.value("B"))

    def test_legacy_t_and_s_fields_are_populated(self, fit):
        """The UI, plots and report all still read result.T and result.S."""
        assert fit.T == fit.value("T")
        assert fit.S == fit.value("S")
        assert fit.T_day == pytest.approx(fit.T * 86400)

    def test_transmissivity_displays_in_m2_per_day(self, fit):
        assert fit.get("T").display_unit == "m^2/day"

    def test_confidence_intervals_are_produced(self, fit):
        for name in ("T", "S", "B"):
            ci = fit.get(name).ci
            assert ci is not None, name
            assert ci[0] < fit.value(name) < ci[1], name

    def test_diagnostics_include_the_steady_state(self, fit):
        assert fit.diagnostics["steady_state_drawdown_m"] > 0
        assert fit.diagnostics["r_over_B"] == pytest.approx(fit.value("r_over_B"))

    def test_state_dict_is_serialisable(self, fit):
        import json
        json.dumps(fit.to_state_dict())

    def test_method_label_matches_the_registry(self, fit):
        from src.calculator import registry
        assert fit.method == registry.get("hantush_jacob_1955").label
        assert fit.method_id == "hantush_jacob_1955"

    def test_goodness_of_fit_carries_aicc(self, fit):
        """Needed the moment a user compares this 3-parameter model against
        2-parameter Theis on the same data."""
        assert fit.gof.n_params == 3
        assert np.isfinite(fit.gof.aicc)


class TestValidityNotes:

    def test_reports_the_predicted_steady_state(self, fit):
        assert any("stabilises" in n for n in fit.validity_notes)

    def test_warns_that_aquitard_storage_is_ignored(self, fit):
        """Hantush (1960) is the model when it is not."""
        assert any("Hantush (1960)" in n for n in fit.validity_notes)

    def test_negligible_leakage_recommends_theis_instead(self):
        """With no detectable leakage the extra parameter is unjustified."""
        df = _synthetic(noise=0.0005, B=1e6)
        result = HantushJacobCalculator().calculate(
            CalculationInput(df=df, Q=Q, r=R))

        if result.value("r_over_B") < NEGLIGIBLE_R_OVER_B:
            assert any("Theis" in n and "fewer parameter" in n
                       for n in result.validity_notes)

    def test_correlation_rule_is_active_for_this_method(self):
        """Three parameters is where non-unique fits start to appear."""
        rules = [r.id for r in HantushJacobCalculator().validity_rules()]
        assert "parameters_correlated" in rules
        assert "too_few_points_per_parameter" in rules


class TestInputValidation:

    def test_too_few_points_is_rejected(self):
        df = _synthetic(n=5)
        result = HantushJacobCalculator().calculate(
            CalculationInput(df=df, Q=Q, r=R))

        assert result.success is False
        assert "Insufficient data" in result.error_message

    def test_negative_drawdown_is_rejected(self):
        df = _synthetic()
        df.loc[0, "drawdown_m"] = -1.0
        result = HantushJacobCalculator().calculate(
            CalculationInput(df=df, Q=Q, r=R))

        assert result.success is False
        assert "Negative drawdown" in result.error_message


# ── The point of the refactor ─────────────────────────────────────────────────

class TestArchitecturePayoff:
    """Evidence that the shared machinery did what it was built for."""

    def test_the_method_contributes_no_fitting_machinery(self):
        """Hantush-Jacob must contain no curve_fit, covariance, CI, goodness-of-fit
        or bound-detection code — all of it is inherited."""
        import inspect
        from src.calculator.methods import hantush_jacob

        source = inspect.getsource(hantush_jacob)
        for forbidden in ("curve_fit", "pcov", "np.diag", "ss_res", "r_squared",
                          "1.96", "maxfev"):
            assert forbidden not in source, (
                f"{forbidden!r} should come from the shared machinery")

    def test_the_numerical_content_is_one_characteristic_root(self):
        import inspect
        from src.calculator.methods import hantush_jacob

        source = inspect.getsource(hantush_jacob)
        assert source.count("q_hantush_jacob") == 2   # the import and the call

    def test_it_inherits_the_general_validity_rules(self):
        rules = {r.id for r in HantushJacobCalculator().validity_rules()}
        assert {"poor_fit", "parameter_at_bound", "parameters_correlated",
                "too_few_points_per_parameter"} <= rules
