"""Cooper-Jacob (1946) straight-line method for confined aquifer pumping test analysis.

Reference: Cooper, H.H. & Jacob, C.E. (1946). A generalized graphical method for
evaluating formation constants and summarizing well-field history. Transactions of
the American Geophysical Union, 27(4), 526-534.

Key equations from the paper (Cooper & Jacob, 1946):
  Equation 7 (approximation valid for small u):
      s ≈ (2.303Q / 4πT) × log₁₀(2.25Tt / r²S)

  When s is plotted against log₁₀(t), a straight line emerges with:
      Δs = slope  (drawdown per log₁₀ cycle)

  Equation 8 (T from slope):
      T = 2.303Q / (4π × Δs)

  Equation 9 (S from t₀, the zero-drawdown time intercept):
      S = 2.25 × T × t₀ / r²

  Validity criterion (paper, p.3):
      "The error [from truncating the series] is tolerable where u is less than
      about 0.02." — Cooper & Jacob (1946)
      u = r²S / (4Tt)
"""

import logging
from typing import Optional

import numpy as np

from src.calculator.base import BaseCalculator, CalculationInput, CalculationResult

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Paper p.3: "tolerable where u is less than about 0.02"
# This is more conservative than the common 0.05 threshold seen in textbooks.
CJ_U_THRESHOLD = 0.02

# Physical bounds (same philosophy as Theis, confined aquifer)
T_MIN, T_MAX = 1e-9, 1.0       # [m²/s]
S_MIN, S_MAX = 1e-9, 1e-2      # [dimensionless]

# Minimum number of valid (u < threshold) points required to fit the straight line
MIN_VALID_POINTS = 4


class CooperJacobCalculator(BaseCalculator):
    """
    Cooper-Jacob (1946) straight-line method for confined aquifer analysis.

    Fits a straight line to the s vs log₁₀(t) plot using only data points
    satisfying the u < 0.02 validity criterion (Cooper & Jacob 1946, p.3).
    Uses a 2-pass iterative approach:
      Pass 1 — regress all data → estimate T, S
      Pass 2 — filter to u < 0.02 using Pass-1 estimates → re-regress

    Aquifer conditions required (from Cooper & Jacob 1946):
      - Confined aquifer (same as Theis)
      - u < 0.02 for all analysed time steps (late-time requirement)
      - Constant pumping rate Q
      - Fully penetrating pumping well

    For early-time data or short tests where u > 0.02, use Theis (1935) instead.
    """

    def __init__(self, u_threshold: float = CJ_U_THRESHOLD):
        """
        Args:
            u_threshold: Maximum allowable u for the straight-line fit.
                         Default 0.02 per Cooper & Jacob (1946) p.3.
                         Can be relaxed to 0.05 per later textbook convention,
                         but this increases truncation error.
        """
        self.u_threshold = u_threshold

    def calculate(self, inputs: CalculationInput) -> CalculationResult:
        """
        Run Cooper-Jacob straight-line analysis.

        Args:
            inputs: CalculationInput with Q in m³/s (not m³/day)

        Returns:
            CalculationResult with T, S, confidence intervals, and validity notes.
            Extra fields in validity_notes describe delta_s, t0, and u compliance.
        """
        error = self._validate(inputs)
        if error:
            return CalculationResult(
                method="Cooper-Jacob (1946)", T=0.0, S=0.0,
                success=False, error_message=error
            )

        time_s, drawdown_m = inputs.get_series()
        Q, r = inputs.Q, inputs.r
        notes = []

        # ── Pass 1: all-data regression for initial T, S estimate ─────────────
        T1, S1, slope1, intercept1 = self._regress(time_s, drawdown_m, Q, r)
        if T1 is None:
            return CalculationResult(
                method="Cooper-Jacob (1946)", T=0.0, S=0.0,
                success=False,
                error_message=(
                    "Pass-1 regression failed: slope is non-positive. "
                    "Ensure drawdown increases with time and data is from the pumping phase."
                )
            )

        # ── Pass 2: filter to u < threshold using Pass-1 estimates ────────────
        u_all = (r ** 2 * S1) / (4.0 * T1 * time_s)
        valid_mask = u_all < self.u_threshold
        n_valid = int(valid_mask.sum())
        n_excluded = len(time_s) - n_valid

        if n_valid < MIN_VALID_POINTS:
            notes.append(
                f"Only {n_valid} point(s) satisfy u < {self.u_threshold} "
                f"({n_excluded} excluded as early-time). "
                f"Cooper-Jacob requires at least {MIN_VALID_POINTS} valid points. "
                "Test duration may be too short — consider using Theis (1935)."
            )
            # Fall back to all-data fit with a warning
            t_fit, s_fit = time_s, drawdown_m
            used_all_data = True
        else:
            t_fit = time_s[valid_mask]
            s_fit = drawdown_m[valid_mask]
            used_all_data = False

        if n_excluded > 0 and not used_all_data:
            notes.append(
                f"{n_excluded} early-time point(s) excluded (u ≥ {self.u_threshold}) — "
                "straight line fitted to late-time data only, as required by Cooper & Jacob (1946)."
            )

        # ── Final regression on filtered data ─────────────────────────────────
        T_fit, S_fit, slope, intercept = self._regress(t_fit, s_fit, Q, r)
        if T_fit is None:
            return CalculationResult(
                method="Cooper-Jacob (1946)", T=0.0, S=0.0,
                success=False,
                error_message=(
                    "Pass-2 regression failed: slope is non-positive on filtered data. "
                    "Check that late-time drawdown is increasing."
                )
            )

        # ── Confidence intervals ───────────────────────────────────────────────
        T_ci, S_ci = self._compute_ci(t_fit, s_fit, slope, intercept, T_fit, S_fit, Q, r, notes)

        # ── Goodness-of-fit on filtered data ──────────────────────────────────
        s_fitted = self._predict(t_fit, slope, intercept)
        residuals = s_fit - s_fitted
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((s_fit - np.mean(s_fit)) ** 2))
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Also compute fitted values on full time range for the plot
        drawdown_fitted_full = self._predict(time_s, slope, intercept)

        # ── Diagnostic values ─────────────────────────────────────────────────
        delta_s = slope   # drawdown per log₁₀ cycle [m/log-cycle]
        # t₀: time at s=0, extrapolated from the fit line: 0 = slope·log₁₀(t₀) + intercept
        # → log₁₀(t₀) = -intercept / slope → t₀ = 10^(-intercept/slope)
        t0 = 10.0 ** (-intercept / slope) if abs(slope) > 1e-12 else None

        # ── Authoritative u check on all used points ───────────────────────────
        u_used = (r ** 2 * S_fit) / (4.0 * T_fit * t_fit)
        u_max_used = float(np.max(u_used))
        if u_max_used >= self.u_threshold:
            notes.append(
                f"Post-fit u check: maximum u on fitted points = {u_max_used:.4f} "
                f"(threshold {self.u_threshold}). Some points violate the validity "
                "criterion — treat T and S as approximate. Theis (1935) is more "
                "appropriate for this dataset."
            )
        else:
            notes.append(
                f"Post-fit u check passed: all {n_valid} fitted points satisfy "
                f"u < {self.u_threshold} (max u = {u_max_used:.4f}). "
                "Cooper-Jacob approximation is valid for this dataset."
            )

        # ── Bounds check ──────────────────────────────────────────────────────
        T_fit = float(np.clip(T_fit, T_MIN, T_MAX))
        S_fit = float(np.clip(S_fit, S_MIN, S_MAX))

        # ── Validity notes ────────────────────────────────────────────────────
        self._add_validity_notes(
            T_fit, S_fit, r_squared, delta_s, t0, t_fit, s_fit, r, notes
        )

        logger.info(
            f"Cooper-Jacob fit: T={T_fit:.3e} m²/s ({T_fit*86400:.1f} m²/day), "
            f"S={S_fit:.3e}, Δs={delta_s:.4f} m/cycle, "
            f"R²={r_squared:.4f}, RMSE={rmse:.4f}m, "
            f"valid_points={n_valid}/{len(time_s)}"
        )

        return CalculationResult(
            method="Cooper-Jacob (1946)",
            T=T_fit,
            S=S_fit,
            T_ci=T_ci,
            S_ci=S_ci,
            rmse=rmse,
            r_squared=r_squared,
            time_s=time_s,
            drawdown_obs=drawdown_m,
            drawdown_fitted=drawdown_fitted_full,
            validity_notes=notes,
            success=True,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _validate(self, inputs: CalculationInput) -> Optional[str]:
        """Return an error string if inputs are invalid, else None."""
        if inputs.Q <= 0:
            return f"Pumping rate Q must be positive (got {inputs.Q}). Ensure Q is in m³/s."
        if inputs.r <= 0:
            return f"Observation distance r must be positive (got {inputs.r} m)."
        if inputs.r < 0.5:
            logger.warning(
                f"r = {inputs.r}m is very small — verify this is the observation well "
                "distance (typically 10–200m), not the pumping well casing radius."
            )
        t, s = inputs.get_series()
        if len(t) < 10:
            return f"Insufficient data: {len(t)} points (minimum 10 required)."
        if np.any(s < 0):
            return "Negative drawdown values detected — remove recovery phase data before fitting."
        if np.any(t <= 0):
            return "Non-positive time values detected — t=0 should be removed by the formatter."
        return None

    def _regress(
        self,
        time_s: np.ndarray,
        drawdown_m: np.ndarray,
        Q: float,
        r: float,
    ) -> tuple:
        """
        Linear regression of drawdown on log₁₀(time).

        Returns (T, S, slope, intercept) or (None, None, None, None) if slope ≤ 0.
        T from equation 8 (Cooper & Jacob 1946): T = 2.303Q / (4π × Δs)
        S from equation 9: S = 2.25 × T × t₀ / r²
        """
        log10_t = np.log10(time_s)
        slope, intercept = np.polyfit(log10_t, drawdown_m, 1)

        if slope <= 0:
            logger.warning(f"Non-positive slope from polyfit: {slope:.4e}")
            return None, None, None, None

        # Equation 8: T = 2.303Q / (4π × Δs)
        T = (2.303 * Q) / (4.0 * np.pi * slope)

        # t₀ = 10^(-intercept / slope)
        t0 = 10.0 ** (-intercept / slope)
        # Equation 9: S = 2.25 × T × t₀ / r²
        S = (2.25 * T * t0) / (r ** 2)

        return T, S, slope, intercept

    @staticmethod
    def _predict(time_s: np.ndarray, slope: float, intercept: float) -> np.ndarray:
        """Evaluate the straight-line model: s = slope × log₁₀(t) + intercept."""
        return slope * np.log10(time_s) + intercept

    def _compute_ci(
        self,
        time_s: np.ndarray,
        drawdown_m: np.ndarray,
        slope: float,
        intercept: float,
        T_fit: float,
        S_fit: float,
        Q: float,
        r: float,
        notes: list,
    ) -> tuple:
        """
        Compute 95% confidence intervals for T and S.

        T = 2.303Q / (4π × slope), so CI on T is asymmetric:
          T_lower = 2.303Q / (4π × slope_upper)
          T_upper = 2.303Q / (4π × slope_lower)

        S = 2.25Tt₀/r², propagated from both slope and intercept uncertainty.
        Uses analytical standard errors from least-squares regression.
        """
        n = len(time_s)
        if n < 3:
            notes.append("Confidence intervals not computed — fewer than 3 valid points.")
            return None, None

        log10_t = np.log10(time_s)
        s_pred = slope * log10_t + intercept
        residuals = drawdown_m - s_pred
        s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))  # residual std error

        # Standard errors for slope and intercept from linear regression formulas
        Sxx = np.sum((log10_t - np.mean(log10_t)) ** 2)
        if Sxx < 1e-14:
            notes.append("Confidence intervals not computed — insufficient time range variance.")
            return None, None

        se_slope = s_err / np.sqrt(Sxx)
        se_intercept = s_err * np.sqrt(np.sum(log10_t ** 2) / (n * Sxx))

        # 95% CI: z = 1.96 for large n; use t-distribution for small samples
        from scipy.stats import t as t_dist
        t_crit = t_dist.ppf(0.975, df=n - 2)

        slope_lower = slope - t_crit * se_slope
        slope_upper = slope + t_crit * se_slope

        # Asymmetric CI on T (inverse relationship with slope)
        if slope_lower <= 0 or slope_upper <= 0:
            notes.append(
                "T confidence interval not computed — slope CI includes non-positive values. "
                "This indicates high uncertainty in the straight-line fit."
            )
            T_ci = None
        else:
            T_lower = (2.303 * Q) / (4.0 * np.pi * slope_upper)
            T_upper = (2.303 * Q) / (4.0 * np.pi * slope_lower)
            T_ci = (T_lower, T_upper)

        # CI on S via error propagation through S = 2.25 × T × 10^(-b/m) / r²
        # where m = slope, b = intercept. Propagate using delta method.
        # S = 2.25 × (2.303Q / 4πm) × 10^(-b/m) / r²
        # ∂S/∂m and ∂S/∂b are complex; approximate with ±1.96 se on b and ±1.96 se on m
        try:
            intercept_lower = intercept - t_crit * se_intercept
            intercept_upper = intercept + t_crit * se_intercept

            # Four corners of (slope, intercept) uncertainty box → take min/max of S
            s_vals = []
            for sl in [slope_lower, slope_upper]:
                if sl <= 0:
                    continue
                for ic in [intercept_lower, intercept_upper]:
                    t0_corner = 10.0 ** (-ic / sl)
                    T_corner = (2.303 * Q) / (4.0 * np.pi * sl)
                    S_corner = (2.25 * T_corner * t0_corner) / (r ** 2)
                    if S_MIN <= S_corner <= S_MAX * 10:  # allow slightly outside bounds for CI
                        s_vals.append(S_corner)

            if s_vals:
                S_ci = (min(s_vals), max(s_vals))
            else:
                S_ci = None
                notes.append("S confidence interval not computed — corner values out of physical range.")
        except Exception:
            S_ci = None

        return T_ci, S_ci

    def _add_validity_notes(
        self,
        T: float,
        S: float,
        r_sq: float,
        delta_s: float,
        t0: Optional[float],
        time_s: np.ndarray,
        drawdown_m: np.ndarray,
        r: float,
        notes: list,
    ):
        """Hydrogeological interpretation notes for the Cooper-Jacob result."""

        # Goodness-of-fit on straight line
        if r_sq < 0.97:
            notes.append(
                f"Moderate linearity (R² = {r_sq:.3f} < 0.97) on the semi-log plot. "
                "A poor straight-line fit suggests the data may not be fully in the "
                "linear Cooper-Jacob regime — consider using Theis (1935) for a full-curve fit."
            )

        # Storativity range check
        if S > 1e-3:
            notes.append(
                f"S = {S:.2e} is above the typical confined aquifer range (10⁻⁵ – 10⁻³). "
                "Verify the aquifer is truly artesian."
            )
        if S < 1e-6:
            notes.append(
                f"S = {S:.2e} is below the typical confined range (10⁻⁵). "
                "Check for leakage or partial penetration."
            )

        # t₀ extrapolation note
        if t0 is not None:
            t_min_data = float(time_s.min())
            if t0 < t_min_data:
                notes.append(
                    f"t₀ = {t0:.1f}s (zero-drawdown intercept) is extrapolated before the "
                    f"earliest measurement ({t_min_data:.0f}s). This is expected when the "
                    "pumping test starts after the straight-line regime is established. "
                    "S is computed from this extrapolated t₀ (Cooper & Jacob 1946, eq. 9)."
                )

        # Theis comparison recommendation
        notes.append(
            "Theis (1935) full-curve fitting provides an independent estimate of T and S "
            "valid across all time steps. Comparing both methods is good practice."
        )

        # Recovery analysis hint
        notes.append(
            "If recovery data is available, Theis (1935) equation 7 and the "
            "Cooper-Jacob recovery method provide an independent T estimate. "
            "Recovery analysis is planned for a future release."
        )
