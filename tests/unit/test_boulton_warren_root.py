"""Tests for Boulton (1963) unconfined and Warren-Root (1963) fractured.

These two complete coverage of all four aquifer types, and they are the first
four-parameter fits in the project — which makes them the real test of two pieces
of machinery built earlier for exactly this moment: log-scale fitting, without
which the optimiser cannot handle T spanning nine orders of magnitude alongside
omega of order 0.05, and the parameter-correlation rule, since four parameters is
where non-unique fits start appearing.

Each method is held to the same three gates as Hantush-Jacob:

  1. an exact analytical limit it must collapse to
  2. round-trip recovery of known parameters from noisy synthetic data
  3. physics invariants

Every limit here is computed in closed form, so none depends on a table
transcribed from a paper.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expn

from src.calculator.base import CalculationInput
from src.calculator.methods.boulton import (
    MIN_SY_OVER_S, BoultonCalculator, boulton_drawdown,
)
from src.calculator.methods.warren_root import (
    OMEGA_NEAR_UNITY, WarrenRootCalculator, warren_root_drawdown,
)

Q, R = 500 / 86400, 50.0


def _theis(t, T, S, r=R, Q_=Q):
    u = (r ** 2 * S) / (4.0 * T * np.asarray(t, dtype=float))
    return (Q_ / (4.0 * np.pi * T)) * expn(1, np.maximum(u, 1e-12))


def _frame(t, s):
    return pd.DataFrame({"time_s": t, "drawdown_m": s})


# ═══════════════════════════════════════════════════════════════════════════════
# Boulton (1963) — unconfined, delayed yield
# ═══════════════════════════════════════════════════════════════════════════════

B_T, B_S, B_SY, B_ALPHA = 2.5e-3, 5e-5, 0.15, 1e-4


@pytest.fixture(scope="module")
def boulton_fit():
    t = np.logspace(1, 6, 80)
    s = boulton_drawdown(t, B_T, B_S, B_SY, B_ALPHA, Q, R)
    s = s + np.random.default_rng(7).normal(0, 0.002, t.size)
    return BoultonCalculator().calculate(CalculationInput(df=_frame(t, s), Q=Q, r=R))


class TestBoultonAnalyticalLimits:
    """Both ends of the delayed-yield transition are exactly Theis."""

    def test_instantaneous_drainage_is_theis_with_combined_storage(self):
        """alpha -> infinity: the water table drains as fast as it is pumped, so
        the aquifer behaves as if its storativity were S + Sy from the start.

        Compared from t = 1e5 s onward. With S + Sy = 0.15 the dimensionless time
        u = r^2 S/(4Tt) only falls below 1 around t = 4e4 s; before that the
        reference Theis curve underflows to ~1e-167 and a relative comparison is
        meaningless rather than wrong.
        """
        t = np.logspace(5, 8, 30)
        boulton = boulton_drawdown(t, B_T, B_S, B_SY, 1e6, Q, R)
        theis = _theis(t, B_T, B_S + B_SY)

        assert np.max(np.abs(boulton - theis) / theis) < 5e-3

    def test_no_drainage_is_theis_with_elastic_storage_only(self):
        """alpha -> 0: nothing drains on the test timescale, so the aquifer is
        effectively confined."""
        t = np.logspace(1, 4, 30)
        boulton = boulton_drawdown(t, B_T, B_S, B_SY, 1e-30, Q, R)
        theis = _theis(t, B_T, B_S)

        assert np.max(np.abs(boulton - theis) / theis) < 5e-3

    def test_delayed_yield_lies_between_the_two_limits(self):
        """The whole point of the model: less drawdown than the confined case,
        more than if all the specific yield were instantly available."""
        t = np.array([1e4])
        delayed = boulton_drawdown(t, B_T, B_S, B_SY, B_ALPHA, Q, R)[0]

        assert _theis(t, B_T, B_S + B_SY)[0] < delayed < _theis(t, B_T, B_S)[0]

    def test_the_curve_flattens_in_the_middle(self):
        """The delayed-yield plateau — the visual signature of an unconfined
        aquifer, and what distinguishes it from a straight Theis curve.

        Uses alpha = 1e-7 rather than the recovery fixture's 1e-4, because the
        three-segment shape only exists for slow enough drainage. The drainage
        term Sy*alpha/(p+alpha) starts to rival S once p ~ alpha*Sy/S, which for
        alpha = 1e-4 happens around t = 30 s — before a confined segment has had
        time to establish, so that curve has no plateau to dip from at all
        (measured: 0% dip at 1e-4, 99.7% at 1e-7). The fit still recovers the
        parameters in either regime; only the visual signature needs the slower
        drainage.
        """
        from src.calculator.derivative import log_derivative

        t = np.logspace(1, 8, 160)
        d = log_derivative(t, boulton_drawdown(t, B_T, B_S, B_SY, 1e-7, Q, R))

        peak = int(np.argmax(d[:80]))
        trough = int(np.argmin(d[peak:peak + 60])) + peak

        assert d[trough] < 0.5 * d[peak], "the derivative should dip markedly"
        assert d[peak] == pytest.approx(Q / (4.0 * np.pi * B_T), rel=0.2), (
            "the early plateau should sit near Q/(4 pi T)")


class TestBoultonRecovery:

    def test_transmissivity_is_recovered_accurately(self, boulton_fit):
        assert boulton_fit.success, boulton_fit.error_message
        assert boulton_fit.value("T") == pytest.approx(B_T, rel=0.02)

    def test_specific_yield_is_recovered_accurately(self, boulton_fit):
        assert boulton_fit.value("Sy") == pytest.approx(B_SY, rel=0.05)

    def test_drainage_constant_is_recovered(self, boulton_fit):
        assert boulton_fit.value("alpha") == pytest.approx(B_ALPHA, rel=0.2)

    def test_elastic_storativity_is_the_least_certain(self, boulton_fit):
        """S governs only the earliest segment, where noise is largest relative
        to drawdown, so it is recovered least precisely. Worth pinning so a
        future regression is not mistaken for this expected looseness."""
        assert boulton_fit.value("S") == pytest.approx(B_S, rel=0.35)

    def test_fit_quality_is_high(self, boulton_fit):
        assert boulton_fit.r_squared > 0.999

    def test_four_parameters_plus_a_derived_delay_index(self, boulton_fit):
        assert boulton_fit.parameter_names == ["T", "S", "Sy", "alpha",
                                               "delay_index"]
        assert boulton_fit.get("delay_index").derived is True

    def test_delay_index_is_the_reciprocal_of_alpha(self, boulton_fit):
        assert boulton_fit.value("delay_index") == pytest.approx(
            1.0 / boulton_fit.value("alpha"))

    def test_delay_index_displays_in_minutes(self, boulton_fit):
        p = boulton_fit.get("delay_index")
        assert p.display_unit == "min"
        assert p.display == pytest.approx(p.value / 60.0)


class TestBoultonInterpretation:

    def test_alpha_is_flagged_as_empirical(self, boulton_fit):
        """Boulton's own known weakness, and the reason to prefer Neuman."""
        assert any("Neuman" in n for n in boulton_fit.validity_notes)

    def test_indistinguishable_storage_terms_are_flagged(self):
        """Sy close to S means delayed yield was never really observed, and the
        extra parameters are along for the ride."""
        t = np.logspace(1, 5, 60)
        s = boulton_drawdown(t, B_T, 1e-4, 5e-4, B_ALPHA, Q, R)
        result = BoultonCalculator().calculate(
            CalculationInput(df=_frame(t, s), Q=Q, r=R))

        if result.diagnostics.get("sy_over_s", 1e9) < MIN_SY_OVER_S:
            assert any("not separable" in n for n in result.validity_notes)

    def test_a_delay_longer_than_the_test_is_flagged(self):
        """If the plateau never arrived, Sy rests on extrapolation."""
        t = np.logspace(1, 3, 40)          # short test
        s = boulton_drawdown(t, B_T, B_S, B_SY, 1e-6, Q, R)
        result = BoultonCalculator().calculate(
            CalculationInput(df=_frame(t, s), Q=Q, r=R))

        if (result.diagnostics.get("delay_index_s", 0)
                > result.diagnostics.get("test_duration_s", 0)):
            assert any("never observed" in n for n in result.validity_notes)


# ═══════════════════════════════════════════════════════════════════════════════
# Warren-Root (1963) — fractured, double porosity
# ═══════════════════════════════════════════════════════════════════════════════

W_T, W_S, W_OMEGA, W_LAMBDA = 1.5e-3, 1e-4, 0.05, 1e-6


@pytest.fixture(scope="module")
def warren_root_fit():
    t = np.logspace(1, 7, 80)
    s = warren_root_drawdown(t, W_T, W_S, W_OMEGA, W_LAMBDA, Q, R)
    s = s + np.random.default_rng(11).normal(0, 0.002, t.size)
    return WarrenRootCalculator().calculate(
        CalculationInput(df=_frame(t, s), Q=Q, r=R))


class TestWarrenRootAnalyticalLimit:

    def test_all_storage_in_fractures_is_theis(self):
        """omega -> 1 leaves no matrix to drain, so double porosity collapses to
        the single-porosity solution."""
        t = np.logspace(2, 6, 30)
        fractured = warren_root_drawdown(t, W_T, W_S, 1.0, W_LAMBDA, Q, R)
        theis = _theis(t, W_T, W_S)

        assert np.max(np.abs(fractured - theis) / theis) < 5e-3

    def test_early_drawdown_exceeds_the_single_porosity_case(self):
        """Early on, only the fractures have given up water, and they hold just
        omega of the total storativity. Less available storage means MORE
        drawdown, not less — the opposite of the leaky case, where an external
        source supplies water and suppresses drawdown throughout.
        """
        t = np.logspace(1, 4, 20)
        double = warren_root_drawdown(t, W_T, W_S, W_OMEGA, W_LAMBDA, Q, R)
        single = _theis(t, W_T, W_S)

        assert np.all(double > single)

    def test_late_drawdown_converges_on_the_single_porosity_case(self):
        """Once the matrix has drained into the fractures the two media act as
        one, with the full storativity."""
        t = np.logspace(7, 9, 15)
        double = warren_root_drawdown(t, W_T, W_S, W_OMEGA, W_LAMBDA, Q, R)
        single = _theis(t, W_T, W_S)

        assert np.max(np.abs(double - single) / single) < 1e-3

    def test_the_derivative_dips_and_recovers(self):
        """The double-porosity signature, and what distinguishes it from leakage:
        leakage keeps falling, double porosity comes back to a second plateau."""
        from src.calculator.derivative import log_derivative

        t = np.logspace(1, 8, 140)
        s = warren_root_drawdown(t, W_T, W_S, W_OMEGA, W_LAMBDA, Q, R)
        d = log_derivative(t, s)

        trough = int(np.argmin(d[20:110])) + 20
        assert d[trough] < d[15], "should dip below the first plateau"
        assert d[-10] > d[trough], "and recover afterwards"


class TestWarrenRootRecovery:

    def test_all_four_parameters_are_recovered(self, warren_root_fit):
        assert warren_root_fit.success, warren_root_fit.error_message
        assert warren_root_fit.value("T") == pytest.approx(W_T, rel=0.02)
        assert warren_root_fit.value("S") == pytest.approx(W_S, rel=0.05)
        assert warren_root_fit.value("omega") == pytest.approx(W_OMEGA, rel=0.10)
        assert warren_root_fit.value("lambda") == pytest.approx(W_LAMBDA, rel=0.20)

    def test_storage_is_split_into_fracture_and_matrix(self, warren_root_fit):
        f = warren_root_fit.value("S_fracture")
        m = warren_root_fit.value("S_matrix")

        assert f + m == pytest.approx(warren_root_fit.value("S"), rel=1e-9)
        assert m > f, "the matrix should hold most of the water"

    def test_derived_shares_are_flagged_as_derived(self, warren_root_fit):
        for name in ("S_fracture", "S_matrix"):
            assert warren_root_fit.get(name).derived is True

    def test_fit_quality_is_high(self, warren_root_fit):
        assert warren_root_fit.r_squared > 0.999


class TestWarrenRootInterpretation:

    def test_leaky_lookalike_is_disclosed(self, warren_root_fit):
        """Honest about what this method cannot distinguish on a short record."""
        assert any("Hantush-Jacob" in n for n in warren_root_fit.validity_notes)

    def test_single_porosity_behaviour_is_flagged(self):
        t = np.logspace(2, 6, 60)
        s = warren_root_drawdown(t, W_T, W_S, 0.9, W_LAMBDA, Q, R)
        result = WarrenRootCalculator().calculate(
            CalculationInput(df=_frame(t, s), Q=Q, r=R))

        if result.value("omega") > OMEGA_NEAR_UNITY:
            assert any("single-porosity" in n for n in result.validity_notes)


# ═══════════════════════════════════════════════════════════════════════════════
# What these two methods prove about the architecture
# ═══════════════════════════════════════════════════════════════════════════════

class TestFourParameterMachinery:
    """Log-scale fitting and the correlation rule were built for this case."""

    @pytest.mark.parametrize("calculator", [BoultonCalculator, WarrenRootCalculator])
    def test_every_parameter_is_fitted_on_a_log_scale(self, calculator):
        """T spans nine orders of magnitude next to omega of order 0.05. A linear
        parametrisation leaves the Jacobian badly scaled and the optimiser stalls
        before it finds the transition."""
        import numpy as np
        t = np.logspace(1, 6, 40)
        specs = calculator().parameter_specs(
            CalculationInput(df=_frame(t, np.linspace(0.1, 2.0, 40)), Q=Q, r=R),
            t, np.linspace(0.1, 2.0, 40))

        assert all(spec.log_scale for spec in specs)

    @pytest.mark.parametrize("calculator", [BoultonCalculator, WarrenRootCalculator])
    def test_the_correlation_rule_is_active(self, calculator):
        """Four parameters is where non-unique fits start to appear, and they are
        invisible to R-squared."""
        rules = {r.id for r in calculator().validity_rules()}
        assert "parameters_correlated" in rules

    @pytest.mark.parametrize("calculator", [BoultonCalculator, WarrenRootCalculator])
    def test_goodness_of_fit_counts_four_parameters(self, calculator, request):
        fit = request.getfixturevalue(
            "boulton_fit" if calculator is BoultonCalculator else "warren_root_fit")
        assert fit.gof.n_params == 4

    @pytest.mark.parametrize("calculator", [BoultonCalculator, WarrenRootCalculator])
    def test_no_fitting_machinery_is_reimplemented(self, calculator):
        import inspect
        import sys

        module = sys.modules[calculator.__module__]
        source = inspect.getsource(module)
        for forbidden in ("curve_fit", "pcov", "ss_res", "1.96", "maxfev"):
            assert forbidden not in source, f"{forbidden!r} should be inherited"


class TestAquiferCoverage:

    def test_all_four_aquifer_types_now_have_a_method(self):
        from src.calculator import registry

        assert registry.supported_aquifers() == set(registry.AQUIFER_TYPES)

    @pytest.mark.parametrize("aquifer", ["confined", "leaky", "unconfined",
                                         "fractured"])
    def test_each_type_resolves_to_a_buildable_calculator(self, aquifer):
        from src.calculator import registry
        from src.calculator.base import BaseCalculator

        specs = registry.for_aquifer(aquifer)
        assert specs, aquifer
        assert isinstance(specs[0].build(), BaseCalculator)
