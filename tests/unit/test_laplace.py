"""Tests for the Laplace inversion engine and the solution kernels.

Three layers, in increasing order of what they prove:

1. **Inverter self-tests, physics-free.** Known transform pairs (1/p, 1/p^2,
   1/(p+a), 1/sqrt(p)) checked across six log cycles of t. These establish that
   the inverter is right independently of any hydrogeology, so a later failure
   can be attributed to a method's transform rather than to the machinery.

2. **Analytical degeneracy limits.** Every characteristic root must collapse to
   the Theis root when its extra physics is switched off: Hantush-Jacob as the
   leakage factor grows, Boulton as drainage becomes instantaneous, Warren-Root
   as all storage moves into the fractures, Papadopulos-Cooper as the casing
   radius shrinks. This is the acceptance gate each new method has to pass, and
   it is what proves the shared-kernel factorisation is correct rather than
   merely plausible.

3. **Equivalence with the shipped implementation.** The extracted, vectorised
   inverter must reproduce what the hand-rolled loop produced.

Also pinned: that Stehfest correctly *fails* on an oscillatory transform. That
limitation is why multi-rate schedules must be superposed in the time domain,
and a test that records it stops someone quietly inverting a step function later.
"""

import numpy as np
import pytest
from scipy.special import expn

from src.calculator.laplace import (
    DEFAULT_N, LN2, cross_check_orders, stehfest_coefficients,
    stehfest_invert, stehfest_invert_array, stehfest_nodes,
)
from src.calculator.laplace_solutions import (
    hantush_jacob_transform, papadopulos_cooper_transform, q_boulton,
    q_hantush_1960, q_hantush_jacob, q_theis, q_warren_root,
    radial_line_source, theis_transform, with_wellbore_storage,
)

TIMES = np.logspace(-2, 4, 25)


def _theis_drawdown(t, Q, T, S, r):
    """Closed-form Theis, for comparison against inverted solutions."""
    u = (r ** 2 * S) / (4.0 * T * np.asarray(t, dtype=float))
    return (Q / (4.0 * np.pi * T)) * expn(1, np.maximum(u, 1e-12))


# ── Coefficients ──────────────────────────────────────────────────────────────

class TestStehfestCoefficients:

    def test_default_order_gives_twelve(self):
        assert len(stehfest_coefficients()) == DEFAULT_N

    def test_coefficients_sum_to_zero(self):
        """A property of the Stehfest weights; a useful arithmetic check."""
        assert sum(stehfest_coefficients(12)) == pytest.approx(0.0, abs=1e-6)

    def test_odd_order_is_rejected(self):
        with pytest.raises(ValueError, match="must be even"):
            stehfest_coefficients(11)

    def test_tiny_order_is_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            stehfest_coefficients(0)

    def test_large_order_warns_about_round_off(self, caplog):
        """Beyond ~18 the algorithm gets worse, not better."""
        with caplog.at_level("WARNING"):
            stehfest_coefficients(20)
        assert any("round-off" in r.message for r in caplog.records)

    def test_result_is_immutable_and_cached(self):
        a = stehfest_coefficients(12)
        assert isinstance(a, tuple)
        assert a is stehfest_coefficients(12)


class TestStehfestNodes:

    def test_shape_is_times_by_order(self):
        p, _ = stehfest_nodes(TIMES, 12)
        assert p.shape == (len(TIMES), 12)

    def test_first_node_matches_the_definition(self):
        p, ln2_over_t = stehfest_nodes(np.array([10.0]), 12)
        assert p[0, 0] == pytest.approx(LN2 / 10.0)
        assert ln2_over_t[0] == pytest.approx(LN2 / 10.0)

    def test_non_positive_time_is_rejected(self):
        with pytest.raises(ValueError, match="strictly positive"):
            stehfest_nodes(np.array([0.0, 1.0]))


# ── Layer 1: known transform pairs ────────────────────────────────────────────

class TestKnownTransformPairs:
    """Physics-free correctness of the inverter itself."""

    def test_unit_step(self):
        """F(p) = 1/p  ->  f(t) = 1"""
        got = stehfest_invert_array(lambda p: 1.0 / p, TIMES)
        np.testing.assert_allclose(got, np.ones_like(TIMES), rtol=1e-6)

    def test_ramp(self):
        """F(p) = 1/p^2  ->  f(t) = t"""
        got = stehfest_invert_array(lambda p: 1.0 / p ** 2, TIMES)
        np.testing.assert_allclose(got, TIMES, rtol=1e-6)

    def test_quadratic(self):
        """F(p) = 2/p^3  ->  f(t) = t^2

        The 4.5e-5 tolerance is the measured N=12 accuracy for this transform,
        and it is flat across every decade of t — Stehfest error is scale
        invariant for a power law.
        """
        got = stehfest_invert_array(lambda p: 2.0 / p ** 3, TIMES)
        np.testing.assert_allclose(got, TIMES ** 2, rtol=1e-4)

    @pytest.mark.parametrize("a", [0.1, 1.0, 10.0])
    def test_exponential_decay(self, a):
        """F(p) = 1/(p+a)  ->  f(t) = exp(-a t)

        Restricted to a*t <= 3, where the function is still above ~5e-2. Beyond
        that Stehfest's roughly constant absolute error swamps the decaying
        value — see test_relative_accuracy_degrades_on_a_decaying_function.
        """
        t = np.linspace(0.01, 3.0 / a, 20)
        got = stehfest_invert_array(lambda p: 1.0 / (p + a), t)
        np.testing.assert_allclose(got, np.exp(-a * t), rtol=5e-3)

    def test_inverse_square_root(self):
        """F(p) = 1/sqrt(p)  ->  f(t) = 1/sqrt(pi t). A branch point, which is
        the shape most like the well functions."""
        got = stehfest_invert_array(lambda p: 1.0 / np.sqrt(p), TIMES)
        np.testing.assert_allclose(got, 1.0 / np.sqrt(np.pi * TIMES), rtol=1e-4)

    def test_accuracy_holds_across_six_log_cycles(self):
        """Error on a power law is scale invariant, which is what makes the
        inverter usable over the six decades of time a pumping test spans."""
        t = np.logspace(-3, 3, 40)
        got = stehfest_invert_array(lambda p: 1.0 / p ** 2, t)

        errors = np.abs(got - t) / t
        assert np.max(errors) < 1e-5
        assert np.ptp(errors) < 1e-6, "error should not drift with scale"

    def test_scalar_time_returns_a_float(self):
        got = stehfest_invert_array(lambda p: 1.0 / p, 10.0)
        assert isinstance(got, float)
        assert got == pytest.approx(1.0, rel=1e-6)

    def test_scalar_and_array_paths_agree(self):
        """stehfest_invert loops; stehfest_invert_array vectorises."""
        for t in [0.1, 10.0, 1000.0]:
            looped = stehfest_invert(lambda p: 1.0 / (p + 0.5), t)
            vector = stehfest_invert_array(lambda p: 1.0 / (p + 0.5), t)
            assert looped == pytest.approx(vector, rel=1e-12)

    def test_transform_is_evaluated_once_for_the_whole_grid(self):
        """The vectorisation that makes multi-parameter fits tractable."""
        calls = []

        def counted(p):
            calls.append(np.size(p))
            return 1.0 / p

        stehfest_invert_array(counted, TIMES)
        assert len(calls) == 1
        assert calls[0] == len(TIMES) * DEFAULT_N


class TestKnownLimitations:

    def test_oscillatory_transform_is_not_inverted_correctly(self):
        """Stehfest assumes a smooth transform. sin(at) is the standard
        counter-example, and this is why a multi-rate pumping schedule must be
        superposed in the time domain rather than inverted as a step function."""
        a = 2.0
        t = np.linspace(0.5, 5.0, 20)
        got = stehfest_invert_array(lambda p: a / (p ** 2 + a ** 2), t)

        assert np.max(np.abs(got - np.sin(a * t))) > 0.1

    def test_cross_check_flags_the_oscillatory_case(self):
        a = 2.0
        report = cross_check_orders(lambda p: a / (p ** 2 + a ** 2),
                                    np.linspace(0.5, 5.0, 20))
        assert report["agrees"] is False

    def test_cross_check_passes_for_a_smooth_transform(self):
        """The spread here is the N=8 noise floor (~1.5e-4 on a power law), which
        is three orders below the order-unity disagreement a wrong transform
        produces — that gap is what makes the check useful."""
        report = cross_check_orders(lambda p: 1.0 / p ** 2, TIMES)

        assert report["agrees"] is True
        assert report["max_relative_spread"] < 1e-3

    def test_relative_accuracy_degrades_on_a_decaying_function(self):
        """Recorded deliberately: Stehfest gives roughly constant ABSOLUTE
        accuracy, so a function decaying toward zero loses relative accuracy.

        Drawdown grows logarithmically and is unaffected. Slug-test head
        recovery decays exponentially and is not — which is an independent
        reason the slug methods will need a different inverter.
        """
        t = np.linspace(0.01, 20.0, 40)
        got = stehfest_invert_array(lambda p: 1.0 / (p + 1.0), t)
        exact = np.exp(-t)

        # Absolute accuracy holds...
        assert np.max(np.abs(got - exact)) < 1e-3
        # ...while relative accuracy collapses once the value is tiny.
        assert np.max(np.abs(got - exact) / exact) > 1.0


# ── Layer 2: analytical degeneracy limits ─────────────────────────────────────

Q, T, S, R = 500 / 86400, 2.5e-3, 5e-5, 50.0


class TestTheisKernel:
    """The shared radial line-source kernel must reproduce closed-form Theis."""

    def test_matches_the_closed_form(self):
        t = np.logspace(1, 5, 30)
        inverted = stehfest_invert_array(theis_transform(Q, T, S, R), t)
        closed = _theis_drawdown(t, Q, T, S, R)

        np.testing.assert_allclose(inverted, closed, rtol=1e-4)

    def test_drawdown_is_positive_and_increasing(self):
        t = np.logspace(1, 5, 30)
        s = stehfest_invert_array(theis_transform(Q, T, S, R), t)

        assert np.all(s > 0)
        assert np.all(np.diff(s) > 0)

    def test_drawdown_is_linear_in_discharge(self):
        t = np.logspace(2, 4, 10)
        single = stehfest_invert_array(theis_transform(Q, T, S, R), t)
        double = stehfest_invert_array(theis_transform(2 * Q, T, S, R), t)

        np.testing.assert_allclose(double, 2 * single, rtol=1e-9)

    def test_drawdown_decreases_with_distance(self):
        t = np.array([1000.0])
        near = stehfest_invert_array(theis_transform(Q, T, S, 10.0), t)
        far = stehfest_invert_array(theis_transform(Q, T, S, 200.0), t)

        assert near[0] > far[0]


class TestHantushJacobLimit:
    """Hantush-Jacob (1955) must reduce to Theis when leakage vanishes.

    This is the acceptance gate for the first new method, and the test that
    proves the shared kernel plus a swapped root is a correct factorisation.
    """

    def test_root_reduces_to_theis_for_large_leakage_factor(self):
        p = np.logspace(-4, 2, 20)
        np.testing.assert_allclose(
            q_hantush_jacob(p, T, S, B=1e12), q_theis(p, T, S), rtol=1e-9)

    def test_drawdown_reduces_to_theis_for_large_leakage_factor(self):
        t = np.logspace(1, 5, 25)
        leaky = stehfest_invert_array(hantush_jacob_transform(Q, T, S, R, 1e10), t)
        theis = _theis_drawdown(t, Q, T, S, R)

        assert np.max(np.abs(leaky - theis) / theis) < 5e-3

    def test_leakage_reduces_drawdown(self):
        """Water drawn through the aquitard limits drawdown at late time — the
        physical signature of a leaky aquifer."""
        t = np.logspace(3, 6, 20)
        leaky = stehfest_invert_array(hantush_jacob_transform(Q, T, S, R, 300.0), t)
        theis = _theis_drawdown(t, Q, T, S, R)

        assert np.all(leaky < theis)

    def test_leaky_drawdown_stabilises_at_late_time(self):
        """Hantush-Jacob approaches a steady state; Theis never does."""
        t = np.logspace(4, 8, 30)
        s = stehfest_invert_array(hantush_jacob_transform(Q, T, S, R, 300.0), t)

        late_change = (s[-1] - s[-5]) / s[-1]
        assert abs(late_change) < 1e-3

    def test_stronger_leakage_means_less_drawdown(self):
        t = np.array([1e5])
        weak = stehfest_invert_array(hantush_jacob_transform(Q, T, S, R, 1000.0), t)
        strong = stehfest_invert_array(hantush_jacob_transform(Q, T, S, R, 100.0), t)

        assert strong[0] < weak[0]


class TestBoultonLimit:
    """Boulton (1963) delayed yield."""

    def test_instantaneous_drainage_reduces_to_theis_with_combined_storage(self):
        p = np.logspace(-4, 2, 20)
        Sy = 0.15
        got = q_boulton(p, T, S, Sy, alpha=1e12)
        expected = q_theis(p, T, S + Sy)

        np.testing.assert_allclose(got, expected, rtol=1e-4)

    def test_no_drainage_reduces_to_theis_with_confined_storage(self):
        """alpha -> 0 means the water table never drains on the test timescale."""
        p = np.logspace(-2, 2, 20)
        got = q_boulton(p, T, S, Sy=0.15, alpha=1e-30)

        np.testing.assert_allclose(got, q_theis(p, T, S), rtol=1e-6)

    def test_delayed_yield_lies_between_the_two_limits(self):
        p = np.array([1e-2])
        Sy = 0.15
        delayed = q_boulton(p, T, S, Sy, alpha=1e-3)

        assert q_theis(p, T, S + Sy)[0] <= delayed[0] <= q_theis(p, T, S)[0] or \
               q_theis(p, T, S)[0] <= delayed[0] <= q_theis(p, T, S + Sy)[0]


class TestWarrenRootLimit:
    """Warren & Root (1963) double porosity."""

    def test_all_storage_in_fractures_reduces_to_theis(self):
        p = np.logspace(-4, 2, 20)
        got = q_warren_root(p, T, S, omega=1.0, lam=1e-6)

        np.testing.assert_allclose(got, q_theis(p, T, S), rtol=1e-9)

    def test_partial_fracture_storage_differs_from_theis(self):
        p = np.logspace(-4, 2, 20)
        got = q_warren_root(p, T, S, omega=0.1, lam=1e-6)

        assert not np.allclose(got, q_theis(p, T, S), rtol=1e-3)


class TestHantush1960Limit:
    """Hantush (1960) leaky with aquitard storage."""

    def test_reduces_toward_hantush_jacob_as_aquitard_storage_vanishes(self):
        """With negligible S'_s the aquitard transmits leakage instantaneously,
        which is the Hantush-Jacob assumption."""
        p = np.logspace(-4, 0, 15)
        b_prime, K_prime = 5.0, 1e-8

        got = q_hantush_1960(p, T, S, b_prime, K_prime, Ss_prime=1e-14)
        B = np.sqrt(T * b_prime / K_prime)
        expected = q_hantush_jacob(p, T, S, B)

        np.testing.assert_allclose(got, expected, rtol=1e-3)

    def test_aquitard_storage_delays_the_leakage_response(self):
        p = np.array([1e-2])
        stored = q_hantush_1960(p, T, S, 5.0, 1e-8, Ss_prime=1e-5)
        instant = q_hantush_1960(p, T, S, 5.0, 1e-8, Ss_prime=1e-14)

        assert stored[0] != pytest.approx(instant[0], rel=1e-6)


class TestWellboreStorageKernel:

    def test_vanishing_casing_radius_reduces_to_the_line_source(self):
        """With no casing there is no wellbore storage, so Papadopulos-Cooper
        must collapse onto Theis at r = r_w."""
        t = np.logspace(0, 5, 25)
        r_w = 0.10
        pc = stehfest_invert_array(
            papadopulos_cooper_transform(Q, T, S, r_w, 1e-6), t)
        theis = _theis_drawdown(t, Q, T, S, r_w)

        assert np.max(np.abs(pc - theis) / theis) < 1e-3

    def test_wellbore_storage_suppresses_early_drawdown(self):
        """Early water comes from the casing, not the aquifer."""
        t = np.logspace(0, 2, 15)
        r_w = 0.30
        stored = stehfest_invert_array(
            papadopulos_cooper_transform(Q, T, S, r_w, 0.20), t)
        theis = _theis_drawdown(t, Q, T, S, r_w)

        assert np.all(stored < theis)

    def test_converges_to_theis_at_late_time(self):
        t = np.logspace(6, 8, 10)
        r_w = 0.30
        stored = stehfest_invert_array(
            papadopulos_cooper_transform(Q, T, S, r_w, 0.20), t)
        theis = _theis_drawdown(t, Q, T, S, r_w)

        assert np.max(np.abs(stored - theis) / theis) < 1e-2

    def test_stays_finite_where_plain_bessel_would_underflow(self):
        """The kve substitution: at large p, kv(0,x) and kv(1,x) both underflow
        to zero and the ratio becomes 0/0. kve keeps it finite."""
        value = with_wellbore_storage(1e9, Q, T, 0.30, 0.30, 0.20,
                                      q_theis(1e9, T, S))
        assert np.isfinite(value) and value > 0


# ── Layer 3: equivalence with the shipped implementation ──────────────────────

class TestEquivalenceWithShippedPapadopulosCooper:

    def test_pc_drawdown_matches_the_transform_path(self):
        from src.calculator.papadopulos_cooper import pc_drawdown

        t = np.logspace(1, 4, 30)
        direct = stehfest_invert_array(
            papadopulos_cooper_transform(300 / 86400, 1e-3, 1e-4, 0.30, 0.20), t)
        via_module = pc_drawdown(t, 1e-3, 1e-4, 300 / 86400, 0.30, 0.20)

        np.testing.assert_allclose(via_module, direct, rtol=1e-12)

    def test_scalar_helpers_still_work_for_the_existing_tests(self):
        from src.calculator.papadopulos_cooper import (
            _laplace_sw, _stehfest_sw, _stehfest_v,
        )

        assert len(_stehfest_v(12)) == 12
        assert isinstance(_laplace_sw(0.5, 300 / 86400, 1e-3, 1e-4, 0.3, 0.2), float)
        assert _stehfest_sw(100.0, 300 / 86400, 1e-3, 1e-4, 0.3, 0.2) > 0

    def test_array_and_scalar_laplace_agree(self):
        from src.calculator.papadopulos_cooper import _laplace_sw

        args = (300 / 86400, 1e-3, 1e-4, 0.30, 0.20)
        p = np.array([0.1, 1.0, 10.0])
        arrayed = _laplace_sw(p, *args)

        for i, single in enumerate(p):
            assert _laplace_sw(float(single), *args) == pytest.approx(
                arrayed[i], rel=1e-12)
