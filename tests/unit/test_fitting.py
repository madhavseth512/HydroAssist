"""Unit tests for the shared fitting machinery.

The load-bearing test in this file is
`TestEquivalenceWithExistingCalculators::test_reproduces_the_theis_fit_exactly`.
Step 7 of the migration ports TheisCalculator onto this module, and that port is
only safe if `fit_curve` is numerically identical to the hand-rolled curve_fit
call it replaces. Proving that here, before any calculator changes, is what makes
the port a code move rather than a rewrite.

The rest covers the two capabilities the existing calculators do not have —
log-scale fitting and the information criteria — plus the guards that decide when
an uncertainty estimate should be withheld.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expn

from src.calculator.fitting import (
    DEFAULT_TOL, GoodnessOfFit, ParameterSpec,
    covariance_to_ci, detect_bound_hits, fit_curve, goodness_of_fit,
    parameter_correlations,
)


# ── ParameterSpec ─────────────────────────────────────────────────────────────

class TestParameterSpec:

    def test_rejects_inverted_bounds(self):
        with pytest.raises(ValueError, match="lower bound"):
            ParameterSpec("T", "m^2/s", lower=1.0, upper=1e-9, initial=1e-3)

    def test_log_scale_requires_a_positive_lower_bound(self):
        """log10(0) is undefined, so a log-scale parameter cannot reach zero."""
        with pytest.raises(ValueError, match="positive lower bound"):
            ParameterSpec("T", "m^2/s", lower=0.0, upper=1.0,
                          initial=1e-3, log_scale=True)

    def test_free_by_default_and_fixed_when_pinned(self):
        assert ParameterSpec("T", "m^2/s", 1e-9, 1.0, 1e-3).is_free
        assert not ParameterSpec("T", "m^2/s", 1e-9, 1.0, 1e-3, fixed=2e-3).is_free

    def test_clip_respects_bounds(self):
        spec = ParameterSpec("S", "-", 1e-9, 1e-2, 1e-4)
        assert spec.clip(1.0) == 1e-2
        assert spec.clip(1e-12) == 1e-9
        assert spec.clip(5e-5) == 5e-5


# ── Basic fitting ─────────────────────────────────────────────────────────────

def _linear(x, m, c):
    return m * x + c


LINEAR_SPECS = [
    ParameterSpec("m", "-", -10.0, 10.0, 1.0),
    ParameterSpec("c", "-", -10.0, 10.0, 0.0),
]


class TestFitCurve:

    def test_recovers_known_parameters(self):
        x = np.linspace(0, 10, 50)
        y = 3.0 * x - 2.0
        out = fit_curve(_linear, x, y, LINEAR_SPECS)

        assert out.success
        assert out.values["m"] == pytest.approx(3.0, rel=1e-6)
        assert out.values["c"] == pytest.approx(-2.0, abs=1e-6)

    def test_values_are_keyed_by_parameter_name(self):
        x = np.linspace(0, 10, 50)
        out = fit_curve(_linear, x, 2 * x + 1, LINEAR_SPECS)
        assert set(out.values) == {"m", "c"}

    def test_ordered_matches_spec_order(self):
        x = np.linspace(0, 10, 50)
        out = fit_curve(_linear, x, 2 * x + 1, LINEAR_SPECS)
        assert out.ordered(LINEAR_SPECS) == [out.values["m"], out.values["c"]]

    def test_fixed_parameter_is_held_and_excluded_from_the_free_vector(self):
        x = np.linspace(0, 10, 50)
        y = 3.0 * x - 2.0
        specs = [
            ParameterSpec("m", "-", -10.0, 10.0, 1.0),
            ParameterSpec("c", "-", -10.0, 10.0, 0.0, fixed=-2.0),
        ]
        out = fit_curve(_linear, x, y, specs)

        assert out.values["c"] == -2.0
        assert out.free_names == ["m"]
        assert out.values["m"] == pytest.approx(3.0, rel=1e-6)

    def test_all_parameters_fixed_is_an_error(self):
        specs = [ParameterSpec("m", "-", 0.0, 10.0, 1.0, fixed=3.0),
                 ParameterSpec("c", "-", 0.0, 10.0, 1.0, fixed=1.0)]
        with pytest.raises(ValueError, match="(?i)at least one parameter"):
            fit_curve(_linear, np.linspace(0, 1, 10), np.zeros(10), specs)

    def test_bounds_are_respected(self):
        x = np.linspace(0, 10, 50)
        y = 8.0 * x
        specs = [ParameterSpec("m", "-", 0.0, 2.0, 1.0),
                 ParameterSpec("c", "-", -1.0, 1.0, 0.0)]
        out = fit_curve(_linear, x, y, specs)

        assert out.values["m"] <= 2.0 + 1e-12

    def test_initial_guess_outside_bounds_is_clipped_not_raised(self):
        """curve_fit raises if p0 is out of bounds; the caller should not have to
        care, because an initial estimate is a heuristic."""
        x = np.linspace(0, 10, 50)
        specs = [ParameterSpec("m", "-", 0.0, 5.0, 999.0),
                 ParameterSpec("c", "-", -5.0, 5.0, -999.0)]
        out = fit_curve(_linear, x, 2 * x + 1, specs)
        assert out.success

    def test_failure_is_returned_not_raised(self):
        """A non-converging fit is a user-facing outcome, not an exception."""
        def broken(x, a, b):
            return np.full_like(x, np.nan)

        out = fit_curve(broken, np.linspace(1, 10, 20), np.ones(20),
                        [ParameterSpec("a", "-", 0.0, 1.0, 0.5),
                         ParameterSpec("b", "-", 0.0, 1.0, 0.5)])
        assert out.success is False
        assert out.message


class TestLogScaleFitting:
    """Fitting log10(value) for parameters spanning many orders of magnitude."""

    def test_log_scale_recovers_the_same_answer(self):
        x = np.linspace(1, 10, 40)
        true = 3.7e-6

        def model(t, a):
            return a * t

        natural = fit_curve(model, x, true * x,
                            [ParameterSpec("a", "-", 1e-12, 1e-2, 1e-6)])
        logged = fit_curve(model, x, true * x,
                           [ParameterSpec("a", "-", 1e-12, 1e-2, 1e-6,
                                          log_scale=True)])

        assert natural.values["a"] == pytest.approx(true, rel=1e-4)
        assert logged.values["a"] == pytest.approx(true, rel=1e-4)

    def test_log_scale_returns_natural_units(self):
        """The caller must never see log space."""
        x = np.linspace(1, 10, 40)

        def model(t, a):
            return a * t

        out = fit_curve(model, x, 2e-5 * x,
                        [ParameterSpec("a", "-", 1e-12, 1e-2, 1e-6, log_scale=True)])
        assert 1e-6 < out.values["a"] < 1e-3

    def test_log_scale_confidence_interval_is_asymmetric(self):
        """A symmetric interval in log space is asymmetric in natural units, which
        is the physically right shape for a positive scale parameter."""
        rng = np.random.default_rng(0)
        x = np.linspace(1, 10, 60)

        def model(t, a):
            return a * t

        y = 1e-5 * x + rng.normal(0, 2e-7, x.size)
        spec = ParameterSpec("a", "-", 1e-12, 1e-2, 1e-6, log_scale=True)
        out = fit_curve(model, x, y, [spec])
        ci = covariance_to_ci(out, [spec])["a"]

        assert ci is not None
        lo, hi = ci
        assert lo < out.values["a"] < hi
        below, above = out.values["a"] - lo, hi - out.values["a"]
        assert above > below, "log-space CI should widen upward in natural units"


# ── Uncertainty ───────────────────────────────────────────────────────────────

class TestConfidenceIntervals:

    def _noisy_fit(self, seed=1):
        rng = np.random.default_rng(seed)
        x = np.linspace(0, 10, 60)
        y = 3.0 * x - 2.0 + rng.normal(0, 0.1, x.size)
        return fit_curve(_linear, x, y, LINEAR_SPECS), LINEAR_SPECS

    def test_interval_brackets_the_estimate(self):
        out, specs = self._noisy_fit()
        for name, ci in covariance_to_ci(out, specs).items():
            assert ci[0] < out.values[name] < ci[1]

    def test_interval_brackets_the_truth(self):
        out, specs = self._noisy_fit()
        ci = covariance_to_ci(out, specs)
        assert ci["m"][0] < 3.0 < ci["m"][1]

    def test_higher_confidence_gives_a_wider_interval(self):
        out, specs = self._noisy_fit()
        narrow = covariance_to_ci(out, specs, confidence=0.68)["m"]
        wide = covariance_to_ci(out, specs, confidence=0.99)["m"]

        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])

    def test_fixed_parameter_has_no_interval(self):
        """A pinned value carries no estimation uncertainty."""
        x = np.linspace(0, 10, 50)
        specs = [ParameterSpec("m", "-", -10.0, 10.0, 1.0),
                 ParameterSpec("c", "-", -10.0, 10.0, 0.0, fixed=-2.0)]
        out = fit_curve(_linear, x, 3 * x - 2, specs)

        assert covariance_to_ci(out, specs)["c"] is None

    def test_failed_fit_yields_no_intervals(self):
        from src.calculator.fitting import FitOutcome
        out = FitOutcome(values={"m": 1.0}, covariance=None,
                         free_names=["m"], success=False)
        assert covariance_to_ci(out, LINEAR_SPECS)["m"] is None

    def test_infinite_covariance_yields_no_intervals(self):
        """curve_fit returns inf when it cannot pin down a unique solution;
        reporting an interval then would be a fabrication."""
        from src.calculator.fitting import FitOutcome
        out = FitOutcome(values={"m": 1.0, "c": 0.0},
                         covariance=np.array([[np.inf, 0.0], [0.0, 1.0]]),
                         free_names=["m", "c"])
        assert covariance_to_ci(out, LINEAR_SPECS)["m"] is None

    def test_negative_variance_yields_no_intervals(self):
        from src.calculator.fitting import FitOutcome
        out = FitOutcome(values={"m": 1.0, "c": 0.0},
                         covariance=np.array([[-1.0, 0.0], [0.0, 1.0]]),
                         free_names=["m", "c"])
        assert covariance_to_ci(out, LINEAR_SPECS)["m"] is None

    def test_t_distribution_is_wider_than_normal_for_small_samples(self):
        rng = np.random.default_rng(3)
        x = np.linspace(0, 10, 8)
        y = 3.0 * x - 2.0 + rng.normal(0, 0.1, x.size)
        out = fit_curve(_linear, x, y, LINEAR_SPECS)

        normal = covariance_to_ci(out, LINEAR_SPECS)["m"]
        student = covariance_to_ci(out, LINEAR_SPECS, use_t=True, n_points=8)["m"]
        assert (student[1] - student[0]) > (normal[1] - normal[0])


class TestParameterCorrelations:

    def test_reports_pairwise_correlation(self):
        from src.calculator.fitting import FitOutcome
        cov = np.array([[4.0, 1.8], [1.8, 1.0]])
        out = FitOutcome(values={"a": 1.0, "b": 2.0}, covariance=cov,
                         free_names=["a", "b"])

        assert parameter_correlations(out)[("a", "b")] == pytest.approx(0.9)

    def test_empty_for_a_single_parameter(self):
        from src.calculator.fitting import FitOutcome
        out = FitOutcome(values={"a": 1.0}, covariance=np.array([[1.0]]),
                         free_names=["a"])
        assert parameter_correlations(out) == {}

    def test_empty_when_covariance_is_unusable(self):
        from src.calculator.fitting import FitOutcome
        out = FitOutcome(values={"a": 1.0, "b": 2.0},
                         covariance=np.array([[np.inf, 0.0], [0.0, 1.0]]),
                         free_names=["a", "b"])
        assert parameter_correlations(out) == {}


class TestBoundDetection:

    def test_detects_an_upper_bound_hit(self):
        x = np.linspace(0, 10, 50)
        specs = [ParameterSpec("m", "-", 0.0, 2.0, 1.0),
                 ParameterSpec("c", "-", -1.0, 1.0, 0.0)]
        out = fit_curve(_linear, x, 8.0 * x, specs)

        assert detect_bound_hits(out, specs)["m"] == "upper"

    def test_reports_none_for_an_interior_solution(self):
        x = np.linspace(0, 10, 50)
        out = fit_curve(_linear, x, 3.0 * x - 2.0, LINEAR_SPECS)
        assert detect_bound_hits(out, LINEAR_SPECS) == {"m": None, "c": None}

    def test_fixed_parameter_is_never_a_bound_hit(self):
        x = np.linspace(0, 10, 50)
        specs = [ParameterSpec("m", "-", 0.0, 10.0, 1.0),
                 ParameterSpec("c", "-", 0.0, 10.0, 1.0, fixed=0.0)]
        out = fit_curve(_linear, x, 3.0 * x, specs)
        assert detect_bound_hits(out, specs)["c"] is None


# ── Goodness of fit ───────────────────────────────────────────────────────────

class TestGoodnessOfFit:

    def test_perfect_fit(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        gof = goodness_of_fit(y, y, n_params=2)

        assert gof.rmse == 0.0
        assert gof.r_squared == 1.0

    def test_matches_the_hand_rolled_formulas(self):
        """Same arithmetic as the block duplicated in all three calculators."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        fit = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        gof = goodness_of_fit(obs, fit, n_params=2)

        residuals = obs - fit
        assert gof.rmse == pytest.approx(np.sqrt(np.mean(residuals ** 2)))
        assert gof.ss_res == pytest.approx(np.sum(residuals ** 2))
        expected_r2 = 1 - np.sum(residuals ** 2) / np.sum((obs - obs.mean()) ** 2)
        assert gof.r_squared == pytest.approx(expected_r2)

    def test_zero_variance_data_yields_zero_r_squared(self):
        """Guard preserved from the existing calculators: constant observations
        make R-squared undefined, not 1."""
        obs = np.array([2.0, 2.0, 2.0, 2.0])
        gof = goodness_of_fit(obs, np.array([2.0, 2.1, 1.9, 2.0]), n_params=2)
        assert gof.r_squared == 0.0

    def test_records_sample_and_parameter_counts(self):
        gof = goodness_of_fit(np.arange(20.0), np.arange(20.0) + 0.1, n_params=3)
        assert gof.n_points == 20 and gof.n_params == 3

    def test_adjusted_r_squared_penalises_parameters(self):
        rng = np.random.default_rng(5)
        obs = np.linspace(0, 10, 30)
        fit = obs + rng.normal(0, 0.3, 30)

        few = goodness_of_fit(obs, fit, n_params=2)
        many = goodness_of_fit(obs, fit, n_params=10)
        assert many.adjusted_r_squared < few.adjusted_r_squared
        assert few.r_squared == many.r_squared, "plain R² ignores parameter count"


class TestInformationCriteria:
    """AICc is what stops the app recommending the most complex model by default."""

    def _same_residuals(self, n_params, n=40, seed=7):
        rng = np.random.default_rng(seed)
        obs = np.linspace(0, 10, n)
        fit = obs + rng.normal(0, 0.2, n)
        return goodness_of_fit(obs, fit, n_params=n_params)

    def test_aic_penalises_extra_parameters_at_equal_fit(self):
        assert self._same_residuals(6).aic > self._same_residuals(2).aic

    def test_aicc_penalises_harder_than_aic(self):
        gof = self._same_residuals(6)
        assert gof.aicc > gof.aic

    def test_aicc_penalty_grows_as_data_shrinks(self):
        small = self._same_residuals(6, n=20)
        large = self._same_residuals(6, n=200)
        assert (small.aicc - small.aic) > (large.aicc - large.aic)

    def test_prefer_aicc_flags_typical_pumping_tests(self):
        """60 points and 6 parameters is n/k = 10, squarely in AICc territory."""
        assert self._same_residuals(6, n=60).prefer_aicc is True
        assert self._same_residuals(2, n=200).prefer_aicc is False

    def test_aicc_undefined_when_parameters_exhaust_the_data(self):
        gof = goodness_of_fit(np.arange(5.0), np.arange(5.0) + 0.1, n_params=4)
        assert gof.aicc_is_reliable is False
        assert np.isinf(gof.aicc)


# ── Equivalence with the shipped calculators ──────────────────────────────────

class TestEquivalenceWithExistingCalculators:
    """Proves the step-7 port is a code move, not a rewrite."""

    def test_reproduces_the_theis_fit_exactly(self):
        from src.calculator.base import CalculationInput
        from src.calculator.theis import (
            S_MAX, S_MIN, T_MAX, T_MIN, TheisCalculator, _estimate_initial_T,
            _make_theis_model,
        )

        df = pd.read_csv("tests/theis_cooper_jacob_sample.csv")
        Q, r = 500 / 86400, 50.0
        t = df["time_s"].to_numpy(float)
        s = df["drawdown_m"].to_numpy(float)

        reference = TheisCalculator().calculate(CalculationInput(df=df, Q=Q, r=r))

        T_init, _ = _estimate_initial_T(t, s, Q, r)
        T_init = float(np.clip(T_init, T_MIN * 10, T_MAX / 10))
        specs = [
            ParameterSpec("T", "m^2/s", T_MIN, T_MAX, T_init),
            ParameterSpec("S", "-", S_MIN, S_MAX, 1e-4),
        ]
        out = fit_curve(_make_theis_model(Q, r), t, s, specs, tol=DEFAULT_TOL)

        assert out.success
        assert out.values["T"] == pytest.approx(reference.T, rel=1e-12)
        assert out.values["S"] == pytest.approx(reference.S, rel=1e-12)

    def test_reproduces_the_theis_confidence_intervals_exactly(self):
        from src.calculator.base import CalculationInput
        from src.calculator.theis import (
            S_MAX, S_MIN, T_MAX, T_MIN, TheisCalculator, _estimate_initial_T,
            _make_theis_model,
        )

        df = pd.read_csv("tests/theis_cooper_jacob_sample.csv")
        Q, r = 500 / 86400, 50.0
        t = df["time_s"].to_numpy(float)
        s = df["drawdown_m"].to_numpy(float)

        reference = TheisCalculator().calculate(CalculationInput(df=df, Q=Q, r=r))
        T_init, _ = _estimate_initial_T(t, s, Q, r)
        specs = [
            ParameterSpec("T", "m^2/s", T_MIN, T_MAX,
                          float(np.clip(T_init, T_MIN * 10, T_MAX / 10))),
            ParameterSpec("S", "-", S_MIN, S_MAX, 1e-4),
        ]
        out = fit_curve(_make_theis_model(Q, r), t, s, specs)
        ci = covariance_to_ci(out, specs)

        assert list(ci["T"]) == pytest.approx(list(reference.T_ci), rel=1e-12)
        assert list(ci["S"]) == pytest.approx(list(reference.S_ci), rel=1e-12)

    def test_reproduces_the_theis_goodness_of_fit_exactly(self):
        from src.calculator.base import CalculationInput
        from src.calculator.theis import TheisCalculator, _make_theis_model

        df = pd.read_csv("tests/theis_cooper_jacob_sample.csv")
        Q, r = 500 / 86400, 50.0
        reference = TheisCalculator().calculate(CalculationInput(df=df, Q=Q, r=r))

        fitted = _make_theis_model(Q, r)(df["time_s"].to_numpy(float),
                                         reference.T, reference.S)
        gof = goodness_of_fit(df["drawdown_m"].to_numpy(float), fitted, n_params=2)

        assert gof.rmse == pytest.approx(reference.rmse, rel=1e-12)
        assert gof.r_squared == pytest.approx(reference.r_squared, rel=1e-12)
