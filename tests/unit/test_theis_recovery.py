"""Theis (1935) recovery — the first test type other than constant-rate pumping.

Three hard gates, as for every method: an analytical limit it must converge to
(equation 7 at late recovery), exact recovery of known parameters from synthetic
data, and the behaviour the paper implies when its assumptions fail (a storage
change moves the intercept, not the slope).
"""

import inspect

import numpy as np
import pandas as pd
import pytest
from scipy.special import expn

from src.agents.method_selection import suggest_method, summarise_result
from src.calculator import registry
from src.calculator.base import CalculationInput
from src.calculator.geometry import PumpingSchedule, RateStep
from src.calculator.methods import theis_recovery
from src.calculator.methods.theis_recovery import (
    SINCE_START, SINCE_STOP, TheisRecoveryCalculator, pumping_duration,
    residual_drawdown,
)
from src.calculator.templates import StraightLineCalculator

T, S, Q, R, T_P = 2.5e-3, 5e-5, 500 / 86400, 50.0, 86400.0
SAMPLE = "tests/theis_recovery_sample.csv"


def _inputs(df, t_p=T_P, clock=SINCE_STOP, r=R, **options):
    return CalculationInput(
        df=df, Q=Q, r=r, test_type=registry.RECOVERY,
        schedule=PumpingSchedule(steps=(RateStep(0.0, Q), RateStep(t_p, 0.0))),
        options={"recovery_clock": clock, **options},
    )


def _exact(t_prime, S_prime=None):
    return pd.DataFrame({"time_s": t_prime,
                         "drawdown_m": residual_drawdown(t_prime, T, S, Q, R, T_P,
                                                         S_prime)})


def _ids(result):
    return [f.id for f in result.findings]


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(SAMPLE)


@pytest.fixture(scope="module")
def fit(sample):
    return TheisRecoveryCalculator().calculate(_inputs(sample))


class TestTheModel:

    def test_residual_drawdown_is_two_superposed_theis_wells(self):
        t = np.logspace(1, 5, 20)

        def theis(time):
            return Q / (4 * np.pi * T) * expn(1, R ** 2 * S / (4 * T * time))

        assert residual_drawdown(t, T, S, Q, R, T_P) == pytest.approx(
            theis(T_P + t) - theis(t), rel=1e-12)

    def test_residual_drawdown_falls_as_recovery_proceeds(self):
        s = residual_drawdown(np.logspace(1, 5, 30), T, S, Q, R, T_P)
        assert np.all(np.diff(s) < 0) and s[-1] > 0

    def test_late_recovery_follows_equation_7(self):
        """The analytical limit: once u' is small the exact solution is the
        straight line of equation 7."""
        t = np.logspace(5, 6, 10)       # u' <= 1.25e-4
        exact = residual_drawdown(t, T, S, Q, R, T_P)
        eq7 = 2.303 * Q / (4 * np.pi * T) * np.log10((T_P + t) / t)
        assert exact == pytest.approx(eq7, rel=1e-2)

    def test_pumping_duration_comes_from_the_schedule(self, sample):
        assert pumping_duration(_inputs(sample)) == T_P
        assert pumping_duration(CalculationInput(df=sample, Q=Q, r=R)) is None


class TestRecoversTransmissivity:

    def test_on_the_bundled_sample(self, fit):
        assert fit.success, fit.error_message
        assert fit.T == pytest.approx(T, rel=0.03)

    def test_exactly_on_noise_free_data(self):
        result = TheisRecoveryCalculator().calculate(_inputs(_exact(np.logspace(3, 6, 40))))

        assert result.T == pytest.approx(T, rel=5e-3)
        assert result.value("S_ratio") == pytest.approx(1.0, rel=0.02)
        assert "storage_ratio_near_one" in _ids(result)

    def test_a_storage_change_moves_the_intercept_not_the_slope(self):
        """S' = 4S: equation 7 puts the zero crossing at t/t' = S/S' = 0.25,
        with the slope — and so T — unchanged."""
        result = TheisRecoveryCalculator().calculate(
            _inputs(_exact(np.logspace(4, 6, 40), S_prime=4 * S)))

        assert result.T == pytest.approx(T, rel=0.01)
        assert result.value("S_ratio") == pytest.approx(0.25, rel=0.05)
        assert "storage_ratio_far_from_one" in _ids(result)

    def test_storativity_is_not_reported(self, fit):
        assert fit.get("S") is None
        assert fit.parameter_names == ["T", "S_ratio"]
        assert "storativity_not_determined" in _ids(fit)

    def test_intervals_bracket_the_estimates(self, fit):
        lo, hi = fit.T_ci
        assert lo < fit.T < hi
        lo, hi = fit.get("S_ratio").ci
        assert lo < fit.value("S_ratio") < hi


class TestTheClock:

    def test_counting_from_the_start_gives_the_same_answer(self, sample, fit):
        shifted = sample.assign(time_s=sample.time_s + T_P)
        result = TheisRecoveryCalculator().calculate(_inputs(shifted, clock=SINCE_START))

        assert result.T == pytest.approx(fit.T, rel=1e-9)

    def test_pumping_phase_readings_are_left_out_with_a_note(self, sample, fit):
        pumping = pd.DataFrame({"time_s": np.linspace(600, T_P, 12),
                                "drawdown_m": np.linspace(0.5, 1.2, 12)})
        record = pd.concat([pumping, sample.assign(time_s=sample.time_s + T_P)])
        result = TheisRecoveryCalculator().calculate(_inputs(record, clock=SINCE_START))

        assert result.T == pytest.approx(fit.T, rel=1e-9)
        assert "belong to the pumping phase" in result.validity_notes[0]

    def test_missing_pumping_duration_is_explained(self, sample):
        result = TheisRecoveryCalculator().calculate(
            CalculationInput(df=sample, Q=Q, r=R))
        assert not result.success
        assert "pumping duration" in result.error_message

    def test_too_few_recovery_readings_is_explained(self, sample):
        # The clock is 60 s off, so only 5 of these 15 readings fall after t_p.
        short = sample.head(15).assign(time_s=lambda d: d.time_s + T_P - 60.0)
        result = TheisRecoveryCalculator().calculate(_inputs(short, clock=SINCE_START))

        assert not result.success
        assert "recovery readings remain" in result.error_message

    def test_an_unknown_clock_is_rejected(self, sample):
        result = TheisRecoveryCalculator().calculate(_inputs(sample, clock="sideways"))
        assert "Unknown recovery clock" in result.error_message


class TestPointSelection:

    def test_early_recovery_is_left_out(self, fit):
        d = fit.diagnostics
        assert d["n_excluded"] > 0
        assert d["u_prime_max_fitted"] < theis_recovery.U_THRESHOLD
        assert any("early-recovery reading(s) left out" in n for n in fit.validity_notes)

    def test_an_s_estimate_lets_more_of_the_record_be_used(self, sample, fit):
        result = TheisRecoveryCalculator().calculate(_inputs(sample, S_estimate=S))

        assert result.diagnostics["n_fitted"] > fit.diagnostics["n_fitted"]
        assert result.T == pytest.approx(T, rel=0.03)

    def test_short_recovery_falls_back_to_every_reading(self):
        result = TheisRecoveryCalculator().calculate(_inputs(_exact(np.logspace(1, 3.3, 20))))

        assert result.success
        assert "whole recovery record was used" in result.validity_notes[0]
        assert result.diagnostics["n_excluded"] == 0

    def test_residual_drawdown_below_static_is_not_an_error(self, sample):
        """Late recovery can overshoot static by a millimetre or two of noise;
        the shared drawdown checks would reject that."""
        noisy = sample.copy()
        noisy.loc[noisy.index[-1], "drawdown_m"] = -0.001
        assert TheisRecoveryCalculator().calculate(_inputs(noisy)).success


class TestPlumbing:

    def test_built_on_the_straight_line_template(self):
        assert issubclass(TheisRecoveryCalculator, StraightLineCalculator)
        source = inspect.getsource(theis_recovery)
        for forbidden in ("curve_fit", "np.polyfit", "pcov", "t_dist", "ss_res"):
            assert forbidden not in source, forbidden

    def test_registered_as_the_recovery_method(self):
        spec = registry.get("theis_recovery_1935")
        assert spec.test_types == {registry.RECOVERY}
        assert registry.for_test_type(registry.RECOVERY) == [spec]
        assert registry.canonical_id("the Theis recovery method") == spec.id

    def test_manual_matching_reproduces_the_fit(self, fit):
        from src.calculator.plots import recovery_interactive

        _, at_fit = recovery_interactive(fit, Q, fit.T)
        _, off = recovery_interactive(fit, Q, fit.T * 2)
        assert at_fit == pytest.approx(fit.rmse * 1000, rel=1e-6)
        assert off > at_fit

    def test_the_chat_routes_recovery_data_here(self):
        suggestion = suggest_method(user_message="I have readings from after the "
                                                 "pump was switched off")
        assert suggestion["method"] == "Theis Recovery (1935)"

    def test_the_chat_summary_does_not_invent_a_storativity(self, fit):
        text = summarise_result(fit.to_state_dict())
        assert "not determined" in text
        assert "0.00e+00" not in text
