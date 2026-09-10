"""The three original methods on the shared templates.

Theis and Papadopulos-Cooper moved onto CurveFitCalculator and Cooper-Jacob onto
the new StraightLineCalculator. test_golden_regression.py proves the numbers did
not move; these tests cover what the move was for — that the methods now hold no
fitting machinery of their own, and that they gain what the templates provide:
stable finding ids, published diagnostics, parameter reliability flags and a
goodness-of-fit basis that makes AICc comparable across methods.
"""

import inspect

import numpy as np
import pandas as pd
import pytest
from scipy.special import expn

from src.calculator import cooper_jacob, papadopulos_cooper, theis
from src.calculator.base import CalculationInput
from src.calculator.cooper_jacob import CooperJacobCalculator
from src.calculator.fitting import GoodnessOfFit, ParameterSpec
from src.calculator.papadopulos_cooper import PapadopulosCooperCalculator
from src.calculator.rules import RuleContext, ci_unavailable_rule
from src.calculator.templates import (
    CurveFitCalculator, Selection, StraightLineCalculator,
)
from src.calculator.theis import TheisCalculator

Q_T, R = 0.005787037037037037, 50.0
Q_PC, R_W, R_C = 0.003472222222222222, 0.3, 0.2


@pytest.fixture(scope="module")
def theis_df():
    return pd.read_csv("tests/theis_cooper_jacob_sample.csv")


@pytest.fixture(scope="module")
def theis_result(theis_df):
    return TheisCalculator().calculate(CalculationInput(df=theis_df, Q=Q_T, r=R))


@pytest.fixture(scope="module")
def cj_result(theis_df):
    return CooperJacobCalculator().calculate(CalculationInput(df=theis_df, Q=Q_T, r=R))


@pytest.fixture(scope="module")
def pc_result():
    df = pd.read_csv("tests/papadopulos_cooper_sample.csv")
    return PapadopulosCooperCalculator().calculate(
        CalculationInput(df=df, Q=Q_PC, r=R_W, r_w=R_W, r_c=R_C))


def _ids(result):
    return [f.id for f in result.findings]


class TestOnTheTemplates:

    def test_curve_fit_methods_use_the_curve_fit_template(self):
        assert issubclass(TheisCalculator, CurveFitCalculator)
        assert issubclass(PapadopulosCooperCalculator, CurveFitCalculator)

    def test_cooper_jacob_uses_the_straight_line_template(self):
        assert issubclass(CooperJacobCalculator, StraightLineCalculator)

    @pytest.mark.parametrize("module", [theis, papadopulos_cooper, cooper_jacob],
                             ids=lambda m: m.__name__.rsplit(".", 1)[-1])
    def test_no_hand_rolled_fitting_machinery_remains(self, module):
        """The fixed part lives in the templates. A method reintroducing its own
        optimiser call, covariance handling or residual arithmetic is a copy that
        will drift."""
        source = inspect.getsource(module)
        for forbidden in ("curve_fit", "pcov", "np.diag", "ss_res", "maxfev=",
                          "1.96", "t_dist"):
            assert forbidden not in source, f"{module.__name__} contains {forbidden!r}"

    @pytest.mark.parametrize("fixture,method_id", [
        ("theis_result", "theis_1935"),
        ("cj_result", "cooper_jacob_1946"),
        ("pc_result", "papadopulos_cooper_1967"),
    ])
    def test_results_carry_their_registry_id(self, fixture, method_id, request):
        assert request.getfixturevalue(fixture).method_id == method_id


class TestTheis:

    def test_findings_have_stable_ids(self, theis_result):
        assert _ids(theis_result) == ["cooper_jacob_regime", "recovery_hint"]

    def test_gets_the_shared_rules(self):
        ids = {rule.id for rule in TheisCalculator().validity_rules()}
        assert {"parameters_correlated", "parameter_at_bound",
                "too_few_points_per_parameter",
                "confidence_intervals_unavailable"} <= ids

    def test_carries_goodness_of_fit_with_aicc(self, theis_result):
        assert theis_result.gof is not None
        assert np.isfinite(theis_result.gof.aicc)
        assert theis_result.to_state_dict()["gof"]["basis"] == "fit"

    def test_publishes_the_late_time_u(self, theis_result):
        assert theis_result.diagnostics["u_max_late"] < theis.CJ_U_THRESHOLD

    def test_crude_initial_guess_is_still_reported(self):
        """Short test, far well: the late-time slope is unusable, so the fit
        starts from the fallback and the user must be told."""
        t = np.logspace(1, 2.5, 20)
        u = (200.0 ** 2 * 1e-4) / (4 * 1e-3 * t)
        s = Q_T / (4 * np.pi * 1e-3) * expn(1, u)
        df = pd.DataFrame({"time_s": t, "drawdown_m": s + 1e-4})
        result = TheisCalculator().calculate(CalculationInput(df=df, Q=Q_T, r=200.0))

        assert _ids(result)[0] == "crude_initial_guess"
        assert "Initial parameter guess is crude" in result.validity_notes[0]


class TestPapadopulosCooper:

    def test_storativity_is_flagged_unreliable_and_transmissivity_is_not(self, pc_result):
        assert pc_result.get("S").reliable is False
        assert "p. 244" in pc_result.get("S").reliability_note
        assert pc_result.get("T").reliable is True

    def test_correlation_rule_is_deliberately_absent(self, pc_result):
        """T and S correlate at about -0.99 on this sample while T stays pinned to
        about ±0.2%. The shared rule would call every estimate unreliable; the
        paper-specific S warning is the accurate statement."""
        ids = {rule.id for rule in PapadopulosCooperCalculator().validity_rules()}
        assert "parameters_correlated" not in ids
        assert "storativity_unreliable" in _ids(pc_result)
        assert pc_result.T_ci[1] / pc_result.T_ci[0] < 1.01

    def test_publishes_alpha(self, pc_result):
        assert pc_result.diagnostics["alpha"] == pytest.approx(
            R_W ** 2 * pc_result.S / R_C ** 2)

    def test_findings_keep_the_paper_notes_in_order(self, pc_result):
        assert _ids(pc_result) == [
            "storativity_unreliable", "wellbore_storage_parameter",
            "regime_breakdown", "theis_convergence_time", "compare_with_theis",
        ]


def _theis_series(t, T=2.5e-3, S=5e-5, r=R, Q=Q_T):
    u = (r ** 2 * S) / (4 * T * t)
    return Q / (4 * np.pi * T) * expn(1, u)


def _slope_of(result):
    return np.polyfit(np.log10(result.time_s), result.drawdown_fitted, 1)[0]


class TestCooperJacob:

    def test_findings_start_with_the_point_selection(self, cj_result):
        ids = _ids(cj_result)
        assert ids[0] == "point_selection"
        assert "u_check_passed" in ids
        assert ids[-2:] == ["compare_with_theis", "recovery_hint"]

    def test_publishes_the_line(self, cj_result):
        d = cj_result.diagnostics
        assert d["slope_m_per_log_cycle"] > 0
        assert d["u_max_fitted"] < d["u_threshold"] == cooper_jacob.CJ_U_THRESHOLD
        assert d["n_fitted"] + d["n_excluded"] == len(cj_result.time_s)
        assert d["t0_s"] > 0

    def test_comparison_statistics_use_every_point(self, cj_result):
        """RMSE and R² describe the line on its own points (the golden values);
        AICc must cover the whole record to be comparable with a full-curve fit."""
        assert cj_result.gof is None
        summary = cj_result.to_state_dict()["gof"]
        assert summary["basis"] == "all_points"
        assert summary["n_points"] == len(cj_result.time_s)
        assert summary["n_points"] > cj_result.diagnostics["n_fitted"]

    def test_transmissivity_matches_the_plotted_line(self, cj_result):
        expected = 2.303 * Q_T / (4 * np.pi * _slope_of(cj_result))
        assert cj_result.T == pytest.approx(expected, rel=1e-9)

    def test_fallback_to_all_points_keeps_t_and_the_line_consistent(self):
        """Every point early-time (u > 0.02): the method falls back to all data,
        and the reported T must be the one the plotted line implies."""
        t = np.logspace(1, np.log10(500), 30)
        df = pd.DataFrame({"time_s": t, "drawdown_m": _theis_series(t)})
        result = CooperJacobCalculator().calculate(CalculationInput(df=df, Q=Q_T, r=R))

        assert result.success
        assert "Using all data as fallback" in result.validity_notes[0]
        assert result.T == pytest.approx(2.303 * Q_T / (4 * np.pi * _slope_of(result)),
                                         rel=1e-9)

    def test_threshold_can_be_set_per_run(self, theis_df):
        result = CooperJacobCalculator().calculate(CalculationInput(
            df=theis_df, Q=Q_T, r=R, options={"u_threshold": 0.05}))
        assert result.diagnostics["u_threshold"] == 0.05


class _Line(StraightLineCalculator):
    """The smallest possible straight-line method: the line's own coefficients."""

    method_id = "test_line"
    method_label = "Test line"
    line_failure_message = "The line slopes the wrong way."

    def parameter_specs(self, inputs):
        return [ParameterSpec("a", "m", -10.0, 10.0, 0.0),
                ParameterSpec("b", "m", -10.0, 10.0, 0.0)]

    def line_to_parameters(self, inputs, line):
        return {"a": line.slope, "b": line.intercept} if line.slope > 0 else None


def _line_input(s):
    t = np.logspace(1, 4, 20)
    return CalculationInput(df=pd.DataFrame({"time_s": t, "drawdown_m": s(t)}),
                            Q=1e-3, r=10.0)


class TestStraightLineTemplate:

    def test_recovers_an_exact_line(self):
        result = _Line().calculate(_line_input(lambda t: 0.5 * np.log10(t) + 0.1))

        assert result.success
        assert result.value("a") == pytest.approx(0.5)
        assert result.value("b") == pytest.approx(0.1)
        assert result.r_squared == pytest.approx(1.0)

    def test_a_rejected_line_fails_with_the_method_message(self):
        result = _Line().calculate(_line_input(lambda t: 5.0 - 0.5 * np.log10(t)))

        assert not result.success
        assert result.error_message == "The line slopes the wrong way."

    def test_a_selection_error_stops_the_analysis(self):
        class Refuses(_Line):
            def select_points(self, inputs, t, s):
                return Selection(error="Nothing on the line.")

        result = Refuses().calculate(_line_input(lambda t: np.log10(t)))
        assert result.error_message == "Nothing on the line."

    def test_clipped_values_are_reported_at_their_bound(self):
        result = _Line().calculate(_line_input(lambda t: 20.0 * np.log10(t)))

        assert result.get("a").value == 10.0
        assert result.get("a").at_bound == "upper"
        assert "parameter_at_bound" in _ids(result)

    def test_input_validation_is_shared(self):
        result = _Line().calculate(_line_input(lambda t: -np.ones_like(t)))
        assert "Negative drawdown" in result.error_message


def _context(intervals):
    gof = GoodnessOfFit(rmse=0.0, r_squared=1.0, adjusted_r_squared=1.0,
                        ss_res=0.0, ss_tot=1.0, n_points=20, n_params=2,
                        aic=0.0, aicc=0.0)
    specs = {"T": ParameterSpec("T", "m^2/s", 1e-9, 1.0, 1e-3),
             "S": ParameterSpec("S", "-", 1e-9, 1e-2, 1e-4)}
    return RuleContext(params={"T": 1e-3, "S": 1e-4}, specs=specs, gof=gof,
                       time_s=np.ones(20), drawdown_obs=np.ones(20),
                       drawdown_fitted=np.ones(20), intervals=intervals)


class TestCiUnavailableRule:

    def test_fires_when_no_parameter_has_an_interval(self):
        finding = ci_unavailable_rule("because").evaluate(_context({"T": None, "S": None}))
        assert finding is not None and finding.text == "because"

    def test_silent_when_any_interval_exists(self):
        ctx = _context({"T": (1e-4, 1e-2), "S": None})
        assert ci_unavailable_rule().evaluate(ctx) is None

    def test_silent_when_intervals_were_never_supplied(self):
        """Not supplied is different from could not be computed."""
        assert ci_unavailable_rule().evaluate(_context({})) is None
