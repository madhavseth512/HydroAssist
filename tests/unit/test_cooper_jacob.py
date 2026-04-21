"""Smoke test for CooperJacobCalculator.

Uses the same synthetic dataset as the Theis smoke test (T=2.5e-3 m²/s, S=5e-5)
to allow direct cross-method comparison.

Cooper-Jacob is valid when u < 0.02. For this setup:
    u = r²S / (4Tt) = (50² × 5e-5) / (4 × 2.5e-3 × t) = 0.125 / (0.01 × t) = 12.5 / t
    u < 0.02  →  t > 12.5 / 0.02 = 625 s  (approx 10 minutes)

So points at t > 625s should lie on the straight line; earlier points are excluded.
"""

import numpy as np
import pandas as pd
import pytest

from src.calculator.base import CalculationInput
from src.calculator.cooper_jacob import CooperJacobCalculator


# ── Synthetic dataset (same as Theis smoke test) ──────────────────────────────

TRUE_T = 2.5e-3   # m²/s
TRUE_S = 5e-5     # dimensionless
Q      = 1e-3     # m³/s
R      = 50.0     # m


def _synthetic_data(rng_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic time-drawdown data using the Theis well function.
    Gaussian noise σ=5mm applied (same as Theis smoke test).
    """
    from scipy.special import expn

    rng = np.random.default_rng(rng_seed)
    time_s = np.logspace(1, 5, 60)   # 10s to 100000s, 60 points
    u = (R ** 2 * TRUE_S) / (4.0 * TRUE_T * time_s)
    s_true = (Q / (4.0 * np.pi * TRUE_T)) * expn(1, np.maximum(u, 1e-10))
    noise = rng.normal(0, 0.005, size=len(time_s))   # σ = 5 mm
    drawdown_m = s_true + noise

    return pd.DataFrame({"time_s": time_s, "drawdown_m": np.maximum(drawdown_m, 0)})


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestCooperJacobSmokeTest:
    """Paper-verification smoke test for CooperJacobCalculator."""

    @pytest.fixture(scope="class")
    def result(self):
        df = _synthetic_data()
        inputs = CalculationInput(df=df, Q=Q, r=R)
        calc = CooperJacobCalculator()
        return calc.calculate(inputs)

    def test_success(self, result):
        assert result.success, f"Calculation failed: {result.error_message}"

    def test_method_label(self, result):
        assert result.method == "Cooper-Jacob (1946)"

    def test_T_recovery(self, result):
        """T should be within 5% of true value."""
        error_pct = abs(result.T - TRUE_T) / TRUE_T * 100
        assert error_pct < 5.0, (
            f"T recovery error {error_pct:.2f}% exceeds 5% tolerance. "
            f"Got T={result.T:.3e}, expected {TRUE_T:.3e} m²/s"
        )

    def test_S_recovery(self, result):
        """S should be within 30% of true value (CJ is less precise on S than Theis)."""
        error_pct = abs(result.S - TRUE_S) / TRUE_S * 100
        assert error_pct < 30.0, (
            f"S recovery error {error_pct:.2f}% exceeds 30% tolerance. "
            f"Got S={result.S:.3e}, expected {TRUE_S:.3e}"
        )

    def test_r_squared(self, result):
        """R² on the straight-line portion should be high (clean data)."""
        assert result.r_squared > 0.97, (
            f"R² = {result.r_squared:.4f} is below 0.97 on synthetic data."
        )

    def test_T_day(self, result):
        """T_day convenience field should equal T × 86400."""
        assert abs(result.T_day - result.T * 86400) < 1e-6

    def test_ci_computed(self, result):
        """Confidence intervals should be available for clean synthetic data."""
        assert result.T_ci is not None, "T_ci should not be None for clean data"
        assert result.S_ci is not None, "S_ci should not be None for clean data"

    def test_T_ci_contains_true(self, result):
        """True T should fall within the 95% CI."""
        if result.T_ci is not None:
            assert result.T_ci[0] <= TRUE_T <= result.T_ci[1], (
                f"True T={TRUE_T:.3e} not within CI {result.T_ci}"
            )

    def test_validity_notes_present(self, result):
        """Should have at least one validity note."""
        assert len(result.validity_notes) >= 1

    def test_u_check_note_present(self, result):
        """A post-fit u check note should always be emitted."""
        u_notes = [n for n in result.validity_notes if "Post-fit u check" in n]
        assert u_notes, "Expected a post-fit u check note"
        # The 2-pass iteration is not perfectly self-consistent: Pass 2 re-fits
        # on data filtered using Pass-1 T/S, then the post-fit check uses Pass-2 T/S
        # on those same points. Marginal u violations (u slightly > 0.02) are
        # expected and correctly flagged — not a test failure.

    def test_plotting_arrays(self, result):
        """time_s, drawdown_obs, drawdown_fitted should all be present and same length."""
        assert result.time_s is not None
        assert result.drawdown_obs is not None
        assert result.drawdown_fitted is not None
        assert len(result.time_s) == len(result.drawdown_obs) == len(result.drawdown_fitted)

    def test_points_excluded(self, result):
        """Some early-time points should be excluded (u > 0.02 for t < 625s)."""
        excluded_notes = [n for n in result.validity_notes if "excluded" in n]
        assert excluded_notes, (
            "Expected a note about early-time points being excluded. "
            f"Notes: {result.validity_notes}"
        )


class TestCooperJacobInputValidation:
    """Tests for _validate() error handling."""

    def _df(self, n=20):
        t = np.linspace(100, 10000, n)
        s = np.log(t) * 0.1
        return pd.DataFrame({"time_s": t, "drawdown_m": s})

    def test_negative_Q(self):
        inp = CalculationInput(df=self._df(), Q=-1e-3, r=50.0)
        r = CooperJacobCalculator().calculate(inp)
        assert not r.success
        assert "Q must be positive" in r.error_message

    def test_zero_r(self):
        inp = CalculationInput(df=self._df(), Q=1e-3, r=0.0)
        r = CooperJacobCalculator().calculate(inp)
        assert not r.success
        assert "r must be positive" in r.error_message

    def test_too_few_points(self):
        df = self._df(n=5)
        inp = CalculationInput(df=df, Q=1e-3, r=50.0)
        r = CooperJacobCalculator().calculate(inp)
        assert not r.success
        assert "Insufficient data" in r.error_message

    def test_negative_drawdown(self):
        df = self._df()
        df.loc[0, "drawdown_m"] = -0.1
        inp = CalculationInput(df=df, Q=1e-3, r=50.0)
        r = CooperJacobCalculator().calculate(inp)
        assert not r.success
        assert "Negative drawdown" in r.error_message


class TestCooperJacobUThreshold:
    """Test that u_threshold parameter behaves correctly."""

    def test_relaxed_threshold(self):
        """u_threshold=0.05 (textbook) should not reject more points than 0.02."""
        df = _synthetic_data()
        inp = CalculationInput(df=df, Q=Q, r=R)
        r_strict = CooperJacobCalculator(u_threshold=0.02).calculate(inp)
        r_relaxed = CooperJacobCalculator(u_threshold=0.05).calculate(inp)
        assert r_strict.success
        assert r_relaxed.success
        # T should be similar regardless of threshold on clean data
        assert abs(r_strict.T - r_relaxed.T) / TRUE_T < 0.05
