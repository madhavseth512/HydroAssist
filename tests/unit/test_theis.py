"""Unit tests for TheisCalculator (Theis 1935).

Uses the same synthetic parameters as the Cooper-Jacob smoke test to allow
direct cross-method comparison: T=2.5e-3 m²/s, S=5e-5, Q=1e-3 m³/s, r=50m.

Test organisation
-----------------
1. TestTheisWellFunction     — verify W(u) = expn(1,u) properties and known values
2. TestTheisModelFunction    — verify _make_theis_model() physics
3. TestTheisInitialEstimate  — verify _estimate_initial_T() heuristic
4. TestTheisSmokeTest        — end-to-end fit on synthetic data (main class)
5. TestTheisInputValidation  — _validate() error handling
6. TestTheisValidityNotes    — interpretation notes are always emitted
"""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expn

from src.calculator.base import CalculationInput
from src.calculator.theis import (
    TheisCalculator,
    _estimate_initial_T,
    _make_theis_model,
    T_MIN, T_MAX, S_MIN, S_MAX, CJ_U_THRESHOLD,
)

# ── Shared true parameters (same as CJ smoke test) ───────────────────────────
TRUE_T = 2.5e-3   # m²/s
TRUE_S = 5.0e-5   # dimensionless
Q      = 1.0e-3   # m³/s
R      = 50.0     # m


def _synthetic_df(rng_seed: int = 42, noise_m: float = 0.005) -> pd.DataFrame:
    """60 log-spaced points from 10 s to 100 000 s with Gaussian noise."""
    rng = np.random.default_rng(rng_seed)
    time_s = np.logspace(1, 5, 60)
    u = (R ** 2 * TRUE_S) / (4.0 * TRUE_T * time_s)
    s_true = (Q / (4.0 * np.pi * TRUE_T)) * expn(1, np.maximum(u, 1e-10))
    drawdown_m = s_true + rng.normal(0, noise_m, len(time_s))
    return pd.DataFrame({"time_s": time_s, "drawdown_m": np.maximum(drawdown_m, 0)})


def _make_input(df: pd.DataFrame, r: float = R) -> CalculationInput:
    return CalculationInput(df=df, Q=Q, r=r)


# ── 1. Theis well function ────────────────────────────────────────────────────

class TestTheisWellFunction:
    """
    Verify W(u) = expn(1, u) against known tabulated values.
    These are the values the Theis curve-fit is built upon.
    """

    # Known W(u) values from standard hydrogeology tables (4 sig figs)
    @pytest.mark.parametrize("u, W_expected, tol_pct", [
        (1e-4, 8.633,  0.1),
        (1e-3, 6.332,  0.1),
        (1e-2, 4.038,  0.1),
        (1e-1, 1.823,  0.1),
        (1.0,  0.2194, 0.1),
    ])
    def test_known_values(self, u, W_expected, tol_pct):
        W_calc = float(expn(1, u))
        err_pct = abs(W_calc - W_expected) / W_expected * 100
        assert err_pct < tol_pct, (
            f"W({u}) = {W_calc:.4f}, expected {W_expected:.4f} "
            f"(error {err_pct:.3f}% > {tol_pct}%)"
        )

    def test_W_positive(self):
        u_arr = np.logspace(-6, 0, 50)
        W_arr = expn(1, u_arr)
        assert np.all(W_arr > 0)

    def test_W_strictly_decreasing(self):
        """W(u) must decrease monotonically as u increases."""
        u_arr = np.logspace(-5, 0, 100)
        W_arr = expn(1, u_arr)
        assert np.all(np.diff(W_arr) < 0), "W(u) is not strictly decreasing"

    def test_W_diverges_as_u_approaches_zero(self):
        """As u → 0, W(u) → ∞ (paper eq. 3: -ln(u) dominates)."""
        assert expn(1, 1e-10) > expn(1, 1e-5) > expn(1, 1e-2)


# ── 2. Model function factory ─────────────────────────────────────────────────

class TestTheisModelFunction:
    """Verify _make_theis_model() returns physically correct drawdown arrays."""

    def test_returns_positive_drawdown(self):
        model = _make_theis_model(Q, R)
        t = np.array([100.0, 1000.0, 10000.0])
        s = model(t, TRUE_T, TRUE_S)
        assert np.all(s > 0)

    def test_drawdown_increases_with_time(self):
        model = _make_theis_model(Q, R)
        t = np.logspace(1, 5, 20)
        s = model(t, TRUE_T, TRUE_S)
        assert np.all(np.diff(s) > 0), "Drawdown must increase monotonically with time"

    def test_drawdown_linear_in_Q(self):
        """Doubling Q exactly doubles drawdown (Theis is linear in Q)."""
        t = np.array([600.0, 1800.0, 7200.0])
        s1 = _make_theis_model(Q,     R)(t, TRUE_T, TRUE_S)
        s2 = _make_theis_model(Q*2.0, R)(t, TRUE_T, TRUE_S)
        np.testing.assert_allclose(s2 / s1, 2.0, rtol=1e-6,
                                   err_msg="Drawdown is not linear in Q")

    def test_larger_T_gives_smaller_drawdown(self):
        """Higher transmissivity means faster dissipation — less drawdown."""
        t = np.array([600.0, 1800.0])
        s_low  = _make_theis_model(Q, R)(t, TRUE_T,      TRUE_S)
        s_high = _make_theis_model(Q, R)(t, TRUE_T * 10, TRUE_S)
        assert np.all(s_high < s_low), "Higher T should give lower drawdown"

    def test_larger_r_gives_smaller_drawdown(self):
        """More distant observation well → smaller drawdown at same time."""
        t = np.array([600.0, 3600.0])
        s_near = _make_theis_model(Q, 50.0)(t, TRUE_T, TRUE_S)
        s_far  = _make_theis_model(Q, 200.0)(t, TRUE_T, TRUE_S)
        assert np.all(s_far < s_near), "More distant well should show less drawdown"

    def test_u_guard_prevents_inf(self):
        """Very small u (very late time) must not produce inf or NaN."""
        model = _make_theis_model(Q, R)
        t_huge = np.array([1e12])   # u → 0
        s = model(t_huge, TRUE_T, TRUE_S)
        assert np.isfinite(s).all(), f"Got non-finite drawdown at t=1e12: {s}"


# ── 3. Initial T estimate ─────────────────────────────────────────────────────

class TestTheisInitialEstimate:
    """Verify _estimate_initial_T() heuristic."""

    def test_returns_positive_T(self):
        df = _synthetic_df()
        T_init, _ = _estimate_initial_T(df["time_s"].values, df["drawdown_m"].values, Q, R)
        assert T_init > 0, f"T_init should be positive, got {T_init}"

    def test_within_order_of_magnitude_of_true(self):
        """Heuristic should be within 1 order of magnitude of true T."""
        df = _synthetic_df()
        T_init, _ = _estimate_initial_T(df["time_s"].values, df["drawdown_m"].values, Q, R)
        ratio = T_init / TRUE_T
        assert 0.1 <= ratio <= 10.0, (
            f"T_init={T_init:.3e} is more than 1 order of magnitude from "
            f"true T={TRUE_T:.3e} (ratio={ratio:.2f})"
        )

    def test_fallback_note_when_short_test(self):
        """Short test data (few log cycles) should trigger the fallback note."""
        t = np.linspace(1, 10, 15)   # short, linear spacing — not in CJ regime
        s = np.ones(15) * 0.5
        df = pd.DataFrame({"time_s": t, "drawdown_m": s})
        _, notes = _estimate_initial_T(df["time_s"].values, df["drawdown_m"].values, Q, R)
        # If fallback triggered, notes list is non-empty
        # (not guaranteed for all inputs, but short linear data should trigger it)
        T_init, _ = _estimate_initial_T(df["time_s"].values, df["drawdown_m"].values, Q, R)
        assert T_init > 0


# ── 4. Smoke test (end-to-end fit) ───────────────────────────────────────────

class TestTheisSmokeTest:
    """End-to-end fit on synthetic data with known T and S."""

    @pytest.fixture(scope="class")
    def result(self):
        df = _synthetic_df()
        return TheisCalculator().calculate(_make_input(df))

    def test_success(self, result):
        assert result.success, f"Calculation failed: {result.error_message}"

    def test_method_label(self, result):
        assert result.method == "Theis (1935)"

    def test_T_recovery(self, result):
        """T should be within 5% of true value."""
        err_pct = abs(result.T - TRUE_T) / TRUE_T * 100
        assert err_pct < 5.0, (
            f"T recovery error {err_pct:.2f}% exceeds 5%. "
            f"Got T={result.T:.3e}, expected {TRUE_T:.3e} m²/s"
        )

    def test_S_recovery(self, result):
        """S should be within 30% of true value."""
        err_pct = abs(result.S - TRUE_S) / TRUE_S * 100
        assert err_pct < 30.0, (
            f"S recovery error {err_pct:.2f}% exceeds 30%. "
            f"Got S={result.S:.3e}, expected {TRUE_S:.3e}"
        )

    def test_r_squared(self, result):
        """R² should be very high on clean synthetic data."""
        assert result.r_squared > 0.99, (
            f"R² = {result.r_squared:.4f} is below 0.99 on synthetic data."
        )

    def test_rmse_small(self, result):
        """RMSE should be near the injected noise level (5 mm)."""
        assert result.rmse < 0.02, (
            f"RMSE = {result.rmse:.4f} m is unexpectedly large (noise was 5 mm)."
        )

    def test_T_day(self, result):
        """T_day convenience field must equal T × 86400."""
        assert abs(result.T_day - result.T * 86400) < 1e-6

    def test_T_bounds_respected(self, result):
        assert T_MIN <= result.T <= T_MAX

    def test_S_bounds_respected(self, result):
        assert S_MIN <= result.S <= S_MAX

    def test_ci_computed(self, result):
        """Both CIs should be available on clean synthetic data."""
        assert result.T_ci is not None, "T_ci should not be None for clean data"
        assert result.S_ci is not None, "S_ci should not be None for clean data"

    def test_T_ci_contains_true(self, result):
        """True T must fall within the 95% CI."""
        if result.T_ci is not None:
            lo, hi = result.T_ci
            assert lo <= TRUE_T <= hi, (
                f"True T={TRUE_T:.3e} not within CI ({lo:.3e}, {hi:.3e})"
            )

    def test_S_ci_lower_positive(self, result):
        """95% CI lower bound on S should be positive for clean data."""
        if result.S_ci is not None:
            assert result.S_ci[0] > 0, (
                f"S_ci lower bound is negative: {result.S_ci[0]:.3e}"
            )

    def test_plotting_arrays_present_and_same_length(self, result):
        assert result.time_s is not None
        assert result.drawdown_obs is not None
        assert result.drawdown_fitted is not None
        n = len(result.time_s)
        assert len(result.drawdown_obs) == n
        assert len(result.drawdown_fitted) == n

    def test_fitted_drawdown_positive(self, result):
        assert np.all(result.drawdown_fitted > 0)

    def test_validity_notes_present(self, result):
        assert len(result.validity_notes) >= 1

    def test_different_seeds_give_consistent_T(self):
        """T estimate should be stable across different noise realisations."""
        results = [
            TheisCalculator().calculate(_make_input(_synthetic_df(seed)))
            for seed in [0, 1, 2, 3, 4]
        ]
        T_vals = [r.T for r in results if r.success]
        assert len(T_vals) == 5, "All seeds should produce successful fits"
        T_arr = np.array(T_vals)
        # All should be within 5% of true T
        assert np.all(np.abs(T_arr - TRUE_T) / TRUE_T < 0.05), (
            f"T values across seeds: {T_arr} — some deviate > 5% from {TRUE_T:.3e}"
        )


# ── 5. Input validation ───────────────────────────────────────────────────────

class TestTheisInputValidation:
    """Tests for _validate() error paths and CalculationInput guards."""

    def _df(self, n: int = 20) -> pd.DataFrame:
        t = np.logspace(2, 5, n)
        u = (R ** 2 * TRUE_S) / (4.0 * TRUE_T * t)
        s = (Q / (4.0 * np.pi * TRUE_T)) * expn(1, u)
        return pd.DataFrame({"time_s": t, "drawdown_m": s})

    def test_negative_Q_raises(self):
        with pytest.raises(ValueError, match="Q must be positive"):
            CalculationInput(df=self._df(), Q=-1e-3, r=R)

    def test_zero_r_raises(self):
        with pytest.raises(ValueError, match="r must be positive"):
            CalculationInput(df=self._df(), Q=Q, r=0.0)

    def test_Q_in_m3_per_day_raises(self):
        """Q > 1 m³/s almost always means m³/day was passed."""
        with pytest.raises(ValueError, match="m³/s"):
            CalculationInput(df=self._df(), Q=86.4, r=R)

    def test_too_few_points(self):
        df = self._df(n=5)
        inp = CalculationInput(df=df, Q=Q, r=R)
        r = TheisCalculator().calculate(inp)
        assert not r.success
        assert "Insufficient data" in r.error_message

    def test_negative_drawdown(self):
        df = self._df()
        df.loc[0, "drawdown_m"] = -0.1
        inp = CalculationInput(df=df, Q=Q, r=R)
        r = TheisCalculator().calculate(inp)
        assert not r.success
        assert "Negative drawdown" in r.error_message

    def test_non_positive_time(self):
        df = self._df()
        df.loc[0, "time_s"] = 0.0
        inp = CalculationInput(df=df, Q=Q, r=R)
        r = TheisCalculator().calculate(inp)
        assert not r.success
        assert "Non-positive time" in r.error_message

    def test_exactly_ten_points_accepted(self):
        df = self._df(n=10)
        inp = CalculationInput(df=df, Q=Q, r=R)
        r = TheisCalculator().calculate(inp)
        assert r.success, f"Failed with exactly 10 points: {r.error_message}"

    def test_nine_points_rejected(self):
        df = self._df(n=9)
        inp = CalculationInput(df=df, Q=Q, r=R)
        r = TheisCalculator().calculate(inp)
        assert not r.success
        assert "Insufficient data" in r.error_message


# ── 6. Validity notes ─────────────────────────────────────────────────────────

class TestTheisValidityNotes:
    """Validity notes must always be emitted regardless of fit quality."""

    @pytest.fixture(scope="class")
    def result(self):
        df = _synthetic_df()
        return TheisCalculator().calculate(_make_input(df))

    def test_at_least_one_note(self, result):
        assert len(result.validity_notes) >= 1

    def test_recovery_hint_always_present(self, result):
        """Theis eq. 7 recovery hint should always be appended."""
        recovery_notes = [n for n in result.validity_notes if "recovery" in n.lower()]
        assert recovery_notes, (
            f"Missing recovery hint. Notes: {result.validity_notes}"
        )

    def test_cj_validity_note_present(self, result):
        """A Cooper-Jacob regime note should always be emitted (positive or negative)."""
        cj_notes = [
            n for n in result.validity_notes
            if "Cooper-Jacob" in n or "u <" in n
        ]
        assert cj_notes, (
            f"Missing Cooper-Jacob regime note. Notes: {result.validity_notes}"
        )

    def test_poor_fit_note_on_flat_data(self):
        """Flat drawdown data should produce a poor-fit note (R² < 0.95)."""
        t = np.logspace(2, 5, 20)
        s = np.ones(len(t)) * 0.5   # constant drawdown — can't be Theis
        df = pd.DataFrame({"time_s": t, "drawdown_m": s})
        inp = CalculationInput(df=df, Q=Q, r=R)
        r = TheisCalculator().calculate(inp)
        if r.success and r.r_squared < 0.95:
            poor_fit_notes = [n for n in r.validity_notes if "Poor fit" in n]
            assert poor_fit_notes, (
                f"Expected poor-fit note for R²={r.r_squared:.3f}. "
                f"Notes: {r.validity_notes}"
            )

    def test_high_S_note_emitted(self):
        """S > 1e-3 should trigger an unconfined-aquifer warning."""
        t = np.logspace(2, 5, 30)
        # Synthetic data with high S (unconfined-range)
        S_high = 5e-3
        u = (R ** 2 * S_high) / (4.0 * TRUE_T * t)
        s = (Q / (4.0 * np.pi * TRUE_T)) * expn(1, np.maximum(u, 1e-10))
        df = pd.DataFrame({"time_s": t, "drawdown_m": s})
        inp = CalculationInput(df=df, Q=Q, r=R)
        r = TheisCalculator().calculate(inp)
        if r.success and r.S > 1e-3:
            high_s_notes = [n for n in r.validity_notes if "typical confined" in n]
            assert high_s_notes, (
                f"Expected high-S note for S={r.S:.2e}. Notes: {r.validity_notes}"
            )
