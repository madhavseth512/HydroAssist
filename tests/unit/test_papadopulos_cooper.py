"""Unit tests for PapadopulosCooperCalculator (Papadopulos & Cooper 1967).

Test organisation
-----------------
1. TestStehfestNumerics       — verify the Laplace inversion kernel in isolation
2. TestPhysicalLimits         — check convergence to known analytical limits
3. TestPCSmokeTest            — end-to-end fit on synthetic data (main class)
4. TestPCInputValidation      — _validate() error handling
5. TestPCValidityNotes        — paper-mandated warning notes are always emitted
"""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expn

from src.calculator.base import CalculationInput
from src.calculator.papadopulos_cooper import (
    PapadopulosCooperCalculator,
    _laplace_sw,
    _stehfest_sw,
    _stehfest_v,
    pc_drawdown,
)

# ── Shared true parameters ────────────────────────────────────────────────────
TRUE_T = 1.0e-3   # m²/s
TRUE_S = 1.0e-4   # dimensionless
Q      = 300 / 86400  # m³/s  (300 m³/day)
R_W    = 0.30     # m  — well screen radius
R_C    = 0.20     # m  — casing radius


def _synthetic_df(rng_seed: int = 42) -> pd.DataFrame:
    """50 log-spaced points from 10 s to 10 000 s with 3 mm Gaussian noise."""
    rng = np.random.default_rng(rng_seed)
    t = np.logspace(1, 4, 50)
    sw_true = pc_drawdown(t, TRUE_T, TRUE_S, Q, R_W, R_C)
    noise = rng.normal(0, 0.003, len(t))
    return pd.DataFrame({"time_s": t, "drawdown_m": np.maximum(sw_true + noise, 0)})


def _make_input(df: pd.DataFrame) -> CalculationInput:
    return CalculationInput(df=df, Q=Q, r=R_W, r_w=R_W, r_c=R_C)


# ── 1. Stehfest numerical kernel ─────────────────────────────────────────────

class TestStehfestNumerics:
    """Verify _stehfest_v() and _laplace_sw() in isolation."""

    def test_v_coefficients_length(self):
        V = _stehfest_v(12)
        assert len(V) == 12

    def test_v_coefficients_sum_to_zero(self):
        """Stehfest V coefficients always sum to zero (standard property)."""
        V = _stehfest_v(12)
        assert abs(V.sum()) < 1e-6, f"V.sum() = {V.sum():.2e}, expected ~0"

    def test_v_coefficients_alternating_sign(self):
        """V_i and V_{i+1} have opposite signs (standard property)."""
        V = _stehfest_v(12)
        for i in range(len(V) - 1):
            assert V[i] * V[i + 1] < 0, (
                f"V[{i}]={V[i]:.2e} and V[{i+1}]={V[i+1]:.2e} have same sign"
            )

    def test_laplace_sw_positive(self):
        """Laplace-domain drawdown must be positive for valid inputs."""
        p = 1.0
        val = _laplace_sw(p, Q, TRUE_T, TRUE_S, R_W, R_C)
        assert val > 0

    def test_laplace_sw_near_zero_qrw(self):
        """Returns 0.0 gracefully when q·r_w is vanishingly small."""
        val = _laplace_sw(1e-300, Q, TRUE_T, TRUE_S, R_W, R_C)
        assert val == 0.0

    def test_stehfest_sw_positive(self):
        """Stehfest inversion must give positive drawdown."""
        sw = _stehfest_sw(600.0, Q, TRUE_T, TRUE_S, R_W, R_C)
        assert sw > 0

    def test_stehfest_sw_increases_with_time(self):
        """Drawdown must increase as time increases."""
        t_vals = [60.0, 300.0, 1800.0, 7200.0]
        sw_vals = [_stehfest_sw(t, Q, TRUE_T, TRUE_S, R_W, R_C) for t in t_vals]
        assert sw_vals == sorted(sw_vals), (
            f"Drawdown is not monotonically increasing: {sw_vals}"
        )


# ── 2. Physical limiting behaviour ───────────────────────────────────────────

class TestPhysicalLimits:
    """
    Two analytical limits from the paper:

    Late time  (u_w/α < 10⁻³): F(u_w,α) → W(u_w)  [paper p.242]
    Early time (u_w/α >> 1):    s_w     → Qt/(π·r_c²) [wellbore storage]
    """

    def test_late_time_theis_convergence(self):
        """At u_w/α = 10⁻⁴, F(u_w,α) must match W(u_w) within 1%."""
        # r_w = r_c = 1 → α = S; choose S=0.01
        S, T_unit, r = 0.01, 1.0, 1.0
        alpha = S
        u_w = alpha * 1e-4          # u_w/α = 10⁻⁴ (deep Theis regime)
        t = S / (4.0 * T_unit * u_w)

        sw = _stehfest_sw(t, Q=1.0, T=T_unit, S=S, r_w=r, r_c=r)
        F_calc = sw * 4.0 * np.pi * T_unit / 1.0
        W_uw   = float(expn(1, u_w))

        err_pct = abs(F_calc - W_uw) / W_uw * 100
        assert err_pct < 1.0, (
            f"Late-time Theis error {err_pct:.2f}% > 1%. "
            f"F_calc={F_calc:.4f}, W(u_w)={W_uw:.4f}"
        )

    def test_late_time_theis_convergence_small_alpha(self):
        """Same convergence check for α = 10⁻³ (small α case)."""
        S, T_unit, r = 0.001, 1.0, 1.0
        alpha = S
        u_w = alpha * 1e-4
        t = S / (4.0 * T_unit * u_w)

        sw = _stehfest_sw(t, Q=1.0, T=T_unit, S=S, r_w=r, r_c=r)
        F_calc = sw * 4.0 * np.pi * T_unit / 1.0
        W_uw   = float(expn(1, u_w))

        err_pct = abs(F_calc - W_uw) / W_uw * 100
        assert err_pct < 1.0, (
            f"Late-time Theis error {err_pct:.2f}% > 1% for α=1e-3."
        )

    def test_early_time_wellbore_storage(self):
        """At u_w/α = 50 (wellbore-storage dominated), s_w ≈ Qt/(π·r_c²) within 5%."""
        S, T_unit, r = 0.01, 1.0, 1.0
        alpha = S
        u_w = alpha * 50.0          # u_w/α = 50 (wellbore storage regime)
        t = S / (4.0 * T_unit * u_w)
        Q_unit = 1.0

        sw = _stehfest_sw(t, Q=Q_unit, T=T_unit, S=S, r_w=r, r_c=r)
        sw_ws = Q_unit * t / (np.pi * r**2)

        err_pct = abs(sw - sw_ws) / sw_ws * 100
        assert err_pct < 5.0, (
            f"Early-time wellbore storage error {err_pct:.2f}% > 5%. "
            f"sw_stehfest={sw:.5f}, sw_wellbore={sw_ws:.5f}"
        )

    def test_pc_drawdown_monotonic(self):
        """pc_drawdown must be positive and strictly increasing."""
        t = np.logspace(0, 5, 30)
        sw = pc_drawdown(t, TRUE_T, TRUE_S, Q, R_W, R_C)
        assert np.all(sw > 0), "pc_drawdown returned non-positive values"
        assert np.all(np.diff(sw) > 0), "pc_drawdown is not monotonically increasing"

    def test_pc_drawdown_larger_Q_gives_larger_s(self):
        """Doubling Q should roughly double drawdown (linear in Q)."""
        t = np.array([600.0, 1800.0, 3600.0])
        sw1 = pc_drawdown(t, TRUE_T, TRUE_S, Q,     R_W, R_C)
        sw2 = pc_drawdown(t, TRUE_T, TRUE_S, Q*2.0, R_W, R_C)
        ratios = sw2 / sw1
        assert np.all(np.abs(ratios - 2.0) < 0.01), (
            f"Drawdown not linear in Q: ratios={ratios}"
        )

    def test_pc_drawdown_larger_T_gives_smaller_s(self):
        """Higher T means faster dissipation → less drawdown at same time."""
        t = np.array([600.0, 1800.0])
        sw_low  = pc_drawdown(t, TRUE_T,       TRUE_S, Q, R_W, R_C)
        sw_high = pc_drawdown(t, TRUE_T * 10,  TRUE_S, Q, R_W, R_C)
        assert np.all(sw_high < sw_low), "Higher T should give lower drawdown"


# ── 3. Smoke test (end-to-end fit) ───────────────────────────────────────────

class TestPCSmokeTest:
    """End-to-end fit on synthetic data with known T and S."""

    @pytest.fixture(scope="class")
    def result(self):
        df = _synthetic_df()
        return PapadopulosCooperCalculator().calculate(_make_input(df))

    def test_success(self, result):
        assert result.success, f"Calculation failed: {result.error_message}"

    def test_method_label(self, result):
        assert result.method == "Papadopulos-Cooper (1967)"

    def test_T_recovery(self, result):
        """T should be within 5% of true value (paper: T is the reliable output)."""
        err_pct = abs(result.T - TRUE_T) / TRUE_T * 100
        assert err_pct < 5.0, (
            f"T recovery error {err_pct:.2f}% exceeds 5%. "
            f"Got T={result.T:.3e}, expected {TRUE_T:.3e} m²/s"
        )

    def test_S_recovery(self, result):
        """S should be within one order of magnitude (paper p.244: questionable reliability)."""
        ratio = result.S / TRUE_S
        assert 0.1 <= ratio <= 10.0, (
            f"S ratio {ratio:.2f} outside [0.1, 10]. "
            f"Got S={result.S:.3e}, expected {TRUE_S:.3e}"
        )

    def test_r_squared(self, result):
        """R² should be very high on clean synthetic data."""
        assert result.r_squared > 0.99, (
            f"R² = {result.r_squared:.4f} is below 0.99 on synthetic data."
        )

    def test_rmse_small(self, result):
        """RMSE should be near the injected noise level (3 mm)."""
        assert result.rmse < 0.02, (
            f"RMSE = {result.rmse:.4f} m is unexpectedly large (noise was 3 mm)."
        )

    def test_T_day(self, result):
        """T_day convenience field must equal T × 86400."""
        assert abs(result.T_day - result.T * 86400) < 1e-6

    def test_plotting_arrays_present(self, result):
        """time_s, drawdown_obs, drawdown_fitted must all be present and same length."""
        assert result.time_s is not None
        assert result.drawdown_obs is not None
        assert result.drawdown_fitted is not None
        n = len(result.time_s)
        assert len(result.drawdown_obs) == n
        assert len(result.drawdown_fitted) == n

    def test_plotting_arrays_positive(self, result):
        """Fitted drawdown must be positive."""
        assert np.all(result.drawdown_fitted > 0), "Fitted drawdown contains non-positive values"

    def test_ci_computed(self, result):
        """Both confidence intervals should be available for clean synthetic data."""
        assert result.T_ci is not None, "T_ci should not be None for clean data"
        assert result.S_ci is not None, "S_ci should not be None for clean data"

    def test_T_ci_contains_true(self, result):
        """True T should fall within the 95% CI."""
        if result.T_ci is not None:
            lo, hi = result.T_ci
            assert lo <= TRUE_T <= hi, (
                f"True T={TRUE_T:.3e} not within CI ({lo:.3e}, {hi:.3e})"
            )

    def test_T_bounds_respected(self, result):
        """Fitted T must stay within physical bounds."""
        from src.calculator.papadopulos_cooper import T_MIN, T_MAX
        assert T_MIN <= result.T <= T_MAX

    def test_S_bounds_respected(self, result):
        """Fitted S must stay within physical bounds."""
        from src.calculator.papadopulos_cooper import S_MIN, S_MAX
        assert S_MIN <= result.S <= S_MAX


# ── 4. Input validation ───────────────────────────────────────────────────────

class TestPCInputValidation:
    """Tests for _validate() error paths and CalculationInput guards."""

    def _df(self, n: int = 15) -> pd.DataFrame:
        t = np.logspace(1, 4, n)
        s = np.sqrt(t) * 0.01
        return pd.DataFrame({"time_s": t, "drawdown_m": s})

    def test_negative_Q_raises(self):
        with pytest.raises(ValueError, match="Q must be positive"):
            CalculationInput(df=self._df(), Q=-1e-3, r=R_W, r_w=R_W, r_c=R_C)

    def test_zero_r_w_raises(self):
        with pytest.raises(ValueError, match="r_w must be positive"):
            CalculationInput(df=self._df(), Q=Q, r=R_W, r_w=0.0, r_c=R_C)

    def test_zero_r_c_raises(self):
        with pytest.raises(ValueError, match="r_c must be positive"):
            CalculationInput(df=self._df(), Q=Q, r=R_W, r_w=R_W, r_c=0.0)

    def test_Q_in_m3_per_day_raises(self):
        """Q > 1 m³/s almost always means m³/day was passed — should raise."""
        with pytest.raises(ValueError, match="m³/s"):
            CalculationInput(df=self._df(), Q=300.0, r=R_W, r_w=R_W, r_c=R_C)

    def test_too_few_points(self):
        df = self._df(n=5)
        inp = CalculationInput(df=df, Q=Q, r=R_W, r_w=R_W, r_c=R_C)
        r = PapadopulosCooperCalculator().calculate(inp)
        assert not r.success
        assert "Insufficient data" in r.error_message

    def test_negative_drawdown(self):
        df = self._df()
        df.loc[0, "drawdown_m"] = -0.05
        inp = CalculationInput(df=df, Q=Q, r=R_W, r_w=R_W, r_c=R_C)
        r = PapadopulosCooperCalculator().calculate(inp)
        assert not r.success
        assert "Negative drawdown" in r.error_message

    def test_non_positive_time(self):
        df = self._df()
        df.loc[0, "time_s"] = 0.0
        inp = CalculationInput(df=df, Q=Q, r=R_W, r_w=R_W, r_c=R_C)
        r = PapadopulosCooperCalculator().calculate(inp)
        assert not r.success
        assert "Non-positive time" in r.error_message

    def test_exactly_ten_points_accepted(self):
        """Minimum allowed is 10 points — should succeed on clean data."""
        t = np.logspace(1, 4, 10)
        sw = pc_drawdown(t, TRUE_T, TRUE_S, Q, R_W, R_C)
        df = pd.DataFrame({"time_s": t, "drawdown_m": sw})
        inp = CalculationInput(df=df, Q=Q, r=R_W, r_w=R_W, r_c=R_C)
        r = PapadopulosCooperCalculator().calculate(inp)
        assert r.success, f"Failed with exactly 10 points: {r.error_message}"

    def test_nine_points_rejected(self):
        t = np.logspace(1, 4, 9)
        sw = pc_drawdown(t, TRUE_T, TRUE_S, Q, R_W, R_C)
        df = pd.DataFrame({"time_s": t, "drawdown_m": sw})
        inp = CalculationInput(df=df, Q=Q, r=R_W, r_w=R_W, r_c=R_C)
        r = PapadopulosCooperCalculator().calculate(inp)
        assert not r.success
        assert "Insufficient data" in r.error_message


# ── 5. Validity notes (paper-mandated warnings) ───────────────────────────────

class TestPCValidityNotes:
    """
    Paper p.244 mandates specific user warnings.  These notes must always
    be present in the output regardless of data quality.
    """

    @pytest.fixture(scope="class")
    def result(self):
        df = _synthetic_df()
        return PapadopulosCooperCalculator().calculate(_make_input(df))

    def test_at_least_four_notes(self, result):
        assert len(result.validity_notes) >= 4, (
            f"Expected >=4 notes, got {len(result.validity_notes)}"
        )

    def test_s_reliability_warning_present(self, result):
        """Paper p.244: S has 'questionable reliability' — must always be flagged."""
        s_notes = [n for n in result.validity_notes if "QUESTIONABLE RELIABILITY" in n]
        assert s_notes, (
            "Missing mandatory S reliability warning (paper p.244). "
            f"Notes: {result.validity_notes}"
        )

    def test_alpha_note_present(self, result):
        """The α parameter value and interpretation must be reported."""
        alpha_notes = [n for n in result.validity_notes if "alpha" in n.lower() or "α" in n]
        assert alpha_notes, (
            f"Missing alpha note. Notes: {result.validity_notes}"
        )

    def test_regime_breakdown_note_present(self, result):
        """Fraction of points in each regime must be reported."""
        regime_notes = [n for n in result.validity_notes if "Regime breakdown" in n]
        assert regime_notes, (
            f"Missing regime breakdown note. Notes: {result.validity_notes}"
        )

    def test_convergence_threshold_note_present(self, result):
        """The Theis convergence time threshold must be reported."""
        conv_notes = [n for n in result.validity_notes if "Convergence to Theis" in n]
        assert conv_notes, (
            f"Missing convergence threshold note. Notes: {result.validity_notes}"
        )

    def test_theis_comparison_recommendation(self, result):
        """A recommendation to compare with Theis should always be included."""
        theis_notes = [n for n in result.validity_notes if "Theis" in n and "Run" in n]
        assert theis_notes, (
            f"Missing Theis comparison recommendation. Notes: {result.validity_notes}"
        )
