"""Tests for the Bourdet log-derivative and flow-regime diagnosis.

Two things are being established.

First, that the derivative is numerically right. The decisive check is the Theis
identity: at late time s = (Q/4piT)(ln(4Tt/r^2 S) - gamma), so ds/d ln t settles
at exactly Q/(4 pi T). The height of the flat portion therefore IS the
transmissivity, and can be read off without fitting anything — which is both a
sharp test of the differencing scheme and a genuinely useful independent estimate.

Second, and the reason this module exists at all, that the diagnosis actually
identifies the model that produced the data. Those tests generate curves from the
real Theis, Hantush-Jacob and Papadopulos-Cooper implementations and check that
the derivative points back at the right one. That is what makes it safe to let
`suggest_method` recommend from the data instead of from keywords in a sentence.
"""

import numpy as np
import pytest
from scipy.special import expn

from src.calculator.derivative import (
    DEFAULT_WINDOW, MIN_POINTS, diagnose, log_derivative, loglog_slope,
)
from src.calculator.methods.hantush_jacob import hantush_jacob_drawdown
from src.calculator.papadopulos_cooper import pc_drawdown

Q, R, T_TRUE, S_TRUE = 500 / 86400, 50.0, 2.5e-3, 5e-5


def _theis(t):
    u = (R ** 2 * S_TRUE) / (4.0 * T_TRUE * np.asarray(t, dtype=float))
    return (Q / (4.0 * np.pi * T_TRUE)) * expn(1, np.maximum(u, 1e-12))


# ── Numerical correctness ─────────────────────────────────────────────────────

class TestDerivativeAgainstTheAnalyticalResult:
    """ds/d ln t -> Q/(4 pi T) at late time. The sharpest available check."""

    def test_plateau_equals_Q_over_four_pi_T(self):
        t = np.logspace(3, 7, 60)
        d = log_derivative(t, _theis(t))
        expected = Q / (4.0 * np.pi * T_TRUE)

        assert float(np.median(d[-20:])) == pytest.approx(expected, rel=1e-4)

    def test_plateau_is_flat(self):
        t = np.logspace(4, 7, 40)
        d = log_derivative(t, _theis(t))

        assert float(np.ptp(d[-20:]) / np.median(d[-20:])) < 0.01

    def test_transmissivity_can_be_read_off_without_fitting(self):
        """The practical payoff: an independent T that owes nothing to curve_fit."""
        t = np.logspace(3, 7, 60)
        signature = diagnose(t, _theis(t), Q=Q)

        assert signature.implied_T == pytest.approx(T_TRUE, rel=1e-3)

    def test_derivative_is_independent_of_storativity_at_late_time(self):
        """Q/(4 pi T) contains no S, so the plateau height must not move with it."""
        t = np.logspace(4, 7, 40)

        def theis_with(S):
            u = (R ** 2 * S) / (4.0 * T_TRUE * t)
            return (Q / (4.0 * np.pi * T_TRUE)) * expn(1, np.maximum(u, 1e-12))

        a = float(np.median(log_derivative(t, theis_with(1e-5))[-15:]))
        b = float(np.median(log_derivative(t, theis_with(1e-3))[-15:]))
        assert a == pytest.approx(b, rel=1e-3)


class TestDerivativeMechanics:

    def test_output_length_matches_input(self):
        t = np.logspace(1, 4, 30)
        assert log_derivative(t, _theis(t)).shape == t.shape

    def test_derivative_of_a_pure_log_curve_is_its_coefficient(self):
        """s = a*ln(t) + b  =>  ds/d ln t = a, everywhere."""
        t = np.logspace(0, 5, 50)
        d = log_derivative(t, 0.37 * np.log(t) + 1.2)

        np.testing.assert_allclose(d, 0.37, rtol=1e-9)

    def test_derivative_of_a_constant_is_zero(self):
        t = np.logspace(0, 4, 30)
        np.testing.assert_allclose(log_derivative(t, np.full(30, 2.5)),
                                   0.0, atol=1e-12)

    def test_unsorted_input_is_handled_and_order_preserved(self):
        t = np.logspace(1, 4, 30)
        s = _theis(t)
        shuffle = np.random.default_rng(0).permutation(30)

        shuffled = log_derivative(t[shuffle], s[shuffle])
        straight = log_derivative(t, s)
        np.testing.assert_allclose(shuffled, straight[shuffle], rtol=1e-9)

    def test_a_wider_window_smooths_more(self):
        t = np.logspace(1, 5, 60)
        s = _theis(t) + np.random.default_rng(1).normal(0, 0.002, 60)

        narrow = log_derivative(t, s, window=0.05)
        wide = log_derivative(t, s, window=0.4)
        assert np.std(np.diff(wide)) < np.std(np.diff(narrow))

    def test_survives_noisy_data(self):
        """The whole point of Bourdet's scheme over adjacent differencing."""
        t = np.logspace(3, 7, 80)
        s = _theis(t) + np.random.default_rng(2).normal(0, 0.005, 80)
        d = log_derivative(t, s, window=0.3)

        expected = Q / (4.0 * np.pi * T_TRUE)
        assert float(np.median(d[-30:])) == pytest.approx(expected, rel=0.15)

    @pytest.mark.parametrize("bad,msg", [
        ((np.array([1.0, 2.0]), np.array([1.0, 2.0])), "at least"),
        ((np.logspace(0, 2, 10), np.zeros(9)), "same length"),
    ])
    def test_invalid_input_is_rejected(self, bad, msg):
        with pytest.raises(ValueError, match=msg):
            log_derivative(*bad)

    def test_non_positive_time_is_rejected(self):
        t = np.logspace(0, 3, 10)
        t[0] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            log_derivative(t, np.linspace(1, 2, 10))


class TestLogLogSlope:

    def test_measures_a_power_law_exponent(self):
        x = np.logspace(0, 4, 40)
        assert loglog_slope(x, 3.0 * x ** 0.5) == pytest.approx(0.5, abs=1e-9)

    def test_flat_series_has_zero_slope(self):
        x = np.logspace(0, 4, 40)
        assert loglog_slope(x, np.full(40, 2.0)) == pytest.approx(0.0, abs=1e-9)

    def test_returns_none_without_enough_valid_points(self):
        assert loglog_slope(np.array([1.0, 2.0]), np.array([1.0, -1.0])) is None


# ── Diagnosis: does it identify the model that made the data? ─────────────────

class TestDiagnosisIdentifiesTheSourceModel:
    """Each case is generated by the real calculator for that method."""

    def test_theis_data_reads_as_radial_flow(self):
        t = np.logspace(2, 6, 60)
        signature = diagnose(t, _theis(t), Q=Q)

        assert signature.regime == "radial"
        assert "theis_1935" in signature.suggests
        assert signature.confidence >= 0.7

    def test_leaky_data_reads_as_leakage(self):
        t = np.logspace(2, 7, 60)
        s = hantush_jacob_drawdown(t, T_TRUE, S_TRUE, 300.0, Q, R)
        signature = diagnose(t, s, Q=Q)

        assert signature.regime == "leakage_or_recharge"
        assert "hantush_jacob_1955" in signature.suggests

    def test_wellbore_storage_data_reads_as_wellbore_storage(self):
        t = np.logspace(0, 5, 60)
        s = pc_drawdown(t, 1e-3, 1e-4, 300 / 86400, 0.30, 0.20)
        signature = diagnose(t, s, Q=300 / 86400)

        assert signature.regime == "wellbore_storage"
        assert "papadopulos_cooper_1967" in signature.suggests

    def test_leaky_diagnosis_is_honest_about_the_alternative(self):
        """A constant-head boundary gives the same falling derivative, and the
        tool should say so rather than assert leakage."""
        t = np.logspace(2, 7, 60)
        s = hantush_jacob_drawdown(t, T_TRUE, S_TRUE, 300.0, Q, R)
        signature = diagnose(t, s, Q=Q)

        assert any("boundary" in n for n in signature.notes)

    def test_evidence_is_a_sentence_a_user_can_act_on(self):
        t = np.logspace(2, 6, 60)
        signature = diagnose(t, _theis(t), Q=Q)

        assert "radial flow" in signature.evidence
        assert "m²/day" in signature.evidence, "should quote the implied T"

    def test_linear_flow_is_recognised(self):
        """Half slope throughout — flow into a vertical fracture."""
        t = np.logspace(0, 4, 50)
        signature = diagnose(t, 0.05 * np.sqrt(t), Q=Q)

        assert signature.regime == "linear_flow"
        assert any("not implemented" in n for n in signature.notes)

    def test_rising_derivative_reads_as_a_barrier(self):
        t = np.logspace(1, 5, 50)
        # Radial flow, then the derivative climbs as a no-flow boundary is felt.
        s = 0.2 * np.log(t) + 3e-5 * t
        signature = diagnose(t, s, Q=Q)

        assert signature.regime == "barrier_or_boundary"
        assert signature.suggests == ()
        assert any("biased" in n for n in signature.notes)

    def test_too_few_points_declines_to_guess(self):
        signature = diagnose(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))

        assert signature.regime == "insufficient_data"
        assert signature.confidence == 0.0
        assert str(MIN_POINTS) in signature.evidence

    def test_diagnosis_works_without_a_pumping_rate(self):
        """Q only converts the plateau into T; the regime does not need it."""
        t = np.logspace(2, 6, 60)
        signature = diagnose(t, _theis(t))

        assert signature.regime == "radial"
        assert signature.implied_T is None

    def test_noisy_theis_data_still_reads_as_radial(self):
        t = np.logspace(3, 7, 80)
        s = _theis(t) + np.random.default_rng(3).normal(0, 0.004, 80)
        signature = diagnose(t, s, Q=Q, window=0.3)

        assert signature.regime == "radial"


class TestSignatureShape:

    def test_reports_the_slopes_it_measured(self):
        t = np.logspace(2, 6, 60)
        signature = diagnose(t, _theis(t), Q=Q)

        assert signature.early_slope is not None
        assert signature.late_slope is not None
        assert abs(signature.late_slope) < 0.3, "radial flow means a flat late slope"

    def test_confidence_is_a_probability(self):
        for t, s in [(np.logspace(2, 6, 60), _theis(np.logspace(2, 6, 60)))]:
            signature = diagnose(t, s, Q=Q)
            assert 0.0 <= signature.confidence <= 1.0

    def test_suggested_ids_exist_in_the_registry(self):
        """A suggestion naming a method the registry does not know would break
        the chat layer downstream."""
        from src.calculator import registry

        cases = [
            _theis(np.logspace(2, 6, 60)),
            hantush_jacob_drawdown(np.logspace(2, 7, 60), T_TRUE, S_TRUE,
                                   300.0, Q, R),
        ]
        for s in cases:
            t = np.logspace(2, 6, 60) if len(s) == 60 else np.logspace(2, 7, 60)
            for method_id in diagnose(t, s, Q=Q).suggests:
                assert registry.get(method_id) is not None, method_id
