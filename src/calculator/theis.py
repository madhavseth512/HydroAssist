"""Theis (1935) confined aquifer pumping test calculator.

Reference: Theis, C.V. (1935). The relation between the lowering of the
piezometric surface and the rate and duration of discharge of a well using
ground-water storage. Transactions of the American Geophysical Union, 16, 519-524.

Key equations from the paper:
  Equation 4 (SI form):
      s = (Q / 4πT) × W(u)          where W(u) = ∫[u→∞] (e⁻ᵘ/u) du = expn(1, u)
      u = r²S / (4Tt)

  Equation 3 (series expansion, used to justify u guard and CJ regime):
      W(u) = -0.5772 - ln(u) + u - u²/(2·2!) + u³/(3·3!) - ...
      As u → 0, -ln(u) → +∞ (diverges) → guard required
      For small u, W(u) ≈ -0.5772 - ln(u) → Cooper-Jacob straight-line regime
"""

import logging
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expn

from src.calculator.base import BaseCalculator, CalculationInput, CalculationResult

logger = logging.getLogger(__name__)

# ── Physical bounds for confined aquifer parameters ───────────────────────────
# T_MAX raised to 1.0 m²/s = 86,400 m²/day to cover high-transmissivity
# aquifers (karst, fractured rock). Original plan used 1e-1 which silently
# hits the bound for Snake River Plain-type systems (Fix #3).
T_MIN, T_MAX = 1e-9, 1.0      # [m²/s]

# S_MAX = 1e-2: Theis (1935) explicitly distinguishes compaction-based storage
# (artesian/confined, S = 10⁻⁵–10⁻³) from gravity drainage (unconfined,
# S = 0.01–0.35). Allowing S up to 1.0 would permit unconfined solutions,
# contradicting the method's own assumptions.
S_MIN, S_MAX = 1e-9, 1e-2     # [dimensionless]

# Cooper-Jacob regime threshold — from Theis eq. 3 series truncation
# Align with Cooper-Jacob's default filter threshold (0.02, original paper).
# Using the same value ensures the Theis validity note is consistent with
# what CJ will actually accept when the user runs both methods.
CJ_U_THRESHOLD = 0.02

# curve_fit tolerances tightened from scipy defaults (1e-8) because T and S
# span 4-5 orders of magnitude. Default tolerances can declare convergence
# with a 20% error in T. (Fix #4)
CURVE_FIT_TOL = 1e-12


def _make_theis_model(Q: float, r: float):
    """
    Factory returning the Theis model function with Q and r as closed-over
    invariants. Q and r are fixed test parameters (Theis treats them as
    constants throughout). Signature f(t, T, S) matches curve_fit's requirement.

    Args:
        Q: Pumping rate [m³/s] — MUST be SI before calling
        r: Observation well distance [m]
    """
    def theis_model(t: np.ndarray, T: float, S: float) -> np.ndarray:
        u = (r ** 2 * S) / (4.0 * T * t)
        # Guard: paper eq. 3 shows W(u) = -ln(u) + ... diverges as u → 0.
        # At u = 1e-10, W(u) ≈ 22.4 m — physically extreme but finite. (Fix implicit in plan)
        u = np.maximum(u, 1e-10)
        return (Q / (4.0 * np.pi * T)) * expn(1, u)
    return theis_model


def _estimate_initial_T(time_s: np.ndarray, drawdown_m: np.ndarray,
                         Q: float, r: float) -> tuple[float, list]:
    """
    Estimate T_init from late-time data using the Cooper-Jacob linearisation
    (Theis eq. 3 series truncated for small u: W(u) ≈ -0.5772 - ln(u)).

    Includes a regime pre-check: if late-time u values are likely > 0.5
    (data not yet in linear CJ regime), the CJ slope estimate is unreliable
    and the crude fallback is used immediately. (Fix #1 — partial)

    Returns (T_init, notes_list).
    """
    notes = []

    # Use last third of data (latest time → smallest u → most linear)
    n_late = max(3, len(time_s) // 3)
    t_late = time_s[-n_late:]
    s_late = drawdown_m[-n_late:]

    # Regime pre-check: u = r²S/4Tt. Since T and S are unknown, use the
    # ratio r²/4t (equivalent to T=S=1, which cancels). This is
    # order-of-magnitude neutral and avoids the 1000× error that the
    # canonical T=S=1e-4 proxy produces for high-transmissivity aquifers.
    u_approx = (r ** 2) / (4.0 * t_late)
    in_cj_regime = np.all(u_approx < 0.5)

    if in_cj_regime and len(t_late) >= 3:
        log_t = np.log10(t_late)
        slope, _ = np.polyfit(log_t, s_late, 1)
        if slope > 1e-6:
            T_init = (2.303 * Q) / (4.0 * np.pi * slope)
            logger.debug(f"T_init from CJ slope: {T_init:.3e} m²/s")
            return T_init, notes

    # Fallback: CJ slope method failed (short test or non-linear regime).
    # Setting W(u) = 1 implicitly — crude but within the right order of magnitude.
    # curve_fit with TRF and bounded parameters is robust to this. (Fix #2)
    T_init = Q / (4.0 * np.pi * np.median(drawdown_m))
    notes.append(
        "Initial parameter guess is crude (CJ slope method not applicable — "
        "test duration may be too short for the linear regime). "
        "Verify fit quality carefully."
    )
    logger.warning(f"T_init fallback used: {T_init:.3e} m²/s")
    return T_init, notes


class TheisCalculator(BaseCalculator):
    """
    Theis (1935) pumping test analysis for confined aquifers.

    Fits the Theis well function W(u) to observed time-drawdown data using
    nonlinear least squares (scipy.optimize.curve_fit, Trust Region Reflective).
    Outputs T (transmissivity) and S (storativity) with 95% confidence intervals.

    Aquifer conditions required (from Theis 1935):
      - Confined (artesian) aquifer, bounded above and below by aquitards
      - Homogeneous and isotropic
      - Infinite areal extent (no boundaries within radius of influence)
      - Fully penetrating pumping well
      - Constant pumping rate Q throughout the test
      - Well of infinitesimal diameter (Theis assumption 5)
    """

    def calculate(self, inputs: CalculationInput) -> CalculationResult:
        """
        Run Theis curve fitting.

        Args:
            inputs: CalculationInput with Q in m³/s (not m³/day)

        Returns:
            CalculationResult with T, S, confidence intervals, and validity notes
        """
        # ── Step 1: Validate inputs ───────────────────────────────────────────
        error = self._validate(inputs)
        if error:
            return CalculationResult(
                method="Theis (1935)", T=0.0, S=0.0,
                success=False, error_message=error
            )

        time_s, drawdown_m = inputs.get_series()
        Q, r = inputs.Q, inputs.r
        notes = []

        # ── Step 2: Build model function ──────────────────────────────────────
        theis_model = _make_theis_model(Q, r)

        # ── Step 3: Initial parameter estimates ───────────────────────────────
        T_init, init_notes = _estimate_initial_T(time_s, drawdown_m, Q, r)
        S_init = 1e-4   # canonical midpoint of confined aquifer storativity range
        notes.extend(init_notes)

        # Clamp T_init to bounds
        T_init = np.clip(T_init, T_MIN * 10, T_MAX / 10)

        # ── Step 4: Nonlinear least squares fit ───────────────────────────────
        try:
            popt, pcov = curve_fit(
                theis_model,
                time_s,
                drawdown_m,
                p0=[T_init, S_init],
                bounds=([T_MIN, S_MIN], [T_MAX, S_MAX]),
                maxfev=10000,
                method="trf",       # Trust Region Reflective — handles bounds correctly
                ftol=CURVE_FIT_TOL, # tightened: default 1e-8 too loose for multi-order params
                xtol=CURVE_FIT_TOL,
                gtol=CURVE_FIT_TOL,
            )
        except (RuntimeError, ValueError) as e:
            return CalculationResult(
                method="Theis (1935)", T=0.0, S=0.0,
                success=False,
                error_message=f"Curve fitting failed: {e}. "
                              "Try adjusting Q or r, or check data quality."
            )

        T_fit, S_fit = popt

        # ── Step 5: Confidence intervals with pcov guard ──────────────────────
        T_ci, S_ci = self._compute_ci(pcov, T_fit, S_fit, notes)

        # Detect silent bound-hit on T (Fix #3)
        if abs(T_fit - T_MAX) < 1e-6:
            notes.append(
                f"Transmissivity hit the upper bound ({T_MAX} m²/s = "
                f"{T_MAX * 86400:.0f} m²/day). True T may be higher. "
                "Consider whether this is a high-transmissivity aquifer (karst/fractured rock)."
            )
        if abs(T_fit - T_MIN) < 1e-9:
            notes.append("Transmissivity hit the lower bound — check data quality and Q value.")

        # ── Step 6: Goodness-of-fit ───────────────────────────────────────────
        s_fitted  = theis_model(time_s, T_fit, S_fit)
        residuals = drawdown_m - s_fitted
        rmse      = float(np.sqrt(np.mean(residuals ** 2)))
        ss_res    = float(np.sum(residuals ** 2))
        ss_tot    = float(np.sum((drawdown_m - np.mean(drawdown_m)) ** 2))
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # ── Step 7: Validity notes ────────────────────────────────────────────
        self._add_validity_notes(
            T_fit, S_fit, r_squared, time_s, drawdown_m, Q, r, notes
        )

        logger.info(
            f"Theis fit: T={T_fit:.3e} m²/s ({T_fit*86400:.1f} m²/day), "
            f"S={S_fit:.3e}, R²={r_squared:.4f}, RMSE={rmse:.4f}m"
        )

        return CalculationResult(
            method="Theis (1935)",
            T=float(T_fit),
            S=float(S_fit),
            T_ci=T_ci,
            S_ci=S_ci,
            rmse=rmse,
            r_squared=r_squared,
            time_s=time_s,
            drawdown_obs=drawdown_m,
            drawdown_fitted=s_fitted,
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
            # Very small r likely means the user passed the pumping well casing radius
            # instead of the observation well distance. Flag it.
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

    def _compute_ci(
        self, pcov: np.ndarray, T_fit: float, S_fit: float, notes: list
    ) -> tuple:
        """
        Compute 95% confidence intervals from the curve_fit covariance matrix.
        Returns (T_ci, S_ci) — each is a (lower, upper) tuple or None.

        None is returned when pcov diagonal contains inf or negative values.
        inf → curve_fit did not converge to a unique solution.
        negative → ill-conditioned Jacobian (numerically pathological).
        Both indicate Theis's own warning: heterogeneous aquifer behavior may
        make the theoretical curve a poor model for the data.
        """
        diag = np.diag(pcov)
        if np.any(np.isinf(diag)) or np.any(diag < 0):
            notes.append(
                "Confidence intervals could not be computed — the covariance matrix "
                "is ill-conditioned. This often indicates aquifer heterogeneity or "
                "boundary effects (Theis 1935: 'more observations are always necessary "
                "to guard against heterogeneity of the aquifer')."
            )
            return None, None

        perr = np.sqrt(diag)
        T_ci = (T_fit - 1.96 * perr[0], T_fit + 1.96 * perr[0])
        S_ci = (S_fit - 1.96 * perr[1], S_fit + 1.96 * perr[1])
        return T_ci, S_ci

    def _add_validity_notes(
        self,
        T: float, S: float, r_sq: float,
        time_s: np.ndarray, drawdown_m: np.ndarray,
        Q: float, r: float,
        notes: list,
    ):
        """Generate hydrogeological interpretation notes."""

        # Goodness-of-fit
        if r_sq < 0.95:
            notes.append(
                f"Poor fit (R² = {r_sq:.3f} < 0.95) — the Theis model may not fully "
                "describe this dataset. Possible causes: aquifer heterogeneity, "
                "boundary effects, or leakage (Theis 1935: heterogeneity warning)."
            )

        # Storativity range check (confined aquifer: 10⁻⁵ to 10⁻³)
        if S > 1e-3:
            notes.append(
                f"S = {S:.2e} is above the typical confined aquifer range (10⁻⁵ – 10⁻³). "
                "Verify the aquifer is truly artesian. Higher S suggests unconfined "
                "or leaky conditions, for which Theis assumptions do not hold."
            )

        if S < 1e-6:
            if r < 5.0:
                notes.append(
                    f"S = {S:.2e} is very low. With r = {r}m (short distance), "
                    "storativity is poorly constrained — T is reliable but S is not. "
                    "Use a more distant observation well for a reliable S estimate "
                    "(Theis 1935: time-lag makes S computation unreliable at short r)."
                )
            else:
                notes.append(
                    f"S = {S:.2e} is below the typical confined range (10⁻⁵). "
                    "Check for leakage or partial penetration effects."
                )

        # Cooper-Jacob validity check — correct direction:
        # large t → small u → u < 0.05 → CJ valid (Theis eq. 3 truncation)
        n_late = max(1, len(time_s) // 3)  # Fix #6: max(1,...) prevents -0 slice bug
        u_late = (r ** 2 * S) / (4.0 * T * time_s[-n_late:])
        if np.all(u_late < CJ_U_THRESHOLD):
            notes.append(
                f"u < {CJ_U_THRESHOLD} satisfied for all late-time points "
                "(Cooper-Jacob approximation is valid for this dataset). "
                "Consider also running Cooper-Jacob (1946) for comparison."
            )
        else:
            notes.append(
                "u < 0.05 not satisfied throughout — Theis full-curve fitting "
                "is the appropriate method for this dataset."
            )

        # Recovery data hint (Theis 1935 eq. 7 — Phase 3 feature)
        notes.append(
            "If recovery data is available, Theis (1935) equation 7 provides "
            "an independent T estimate from residual drawdown vs log(t/t'). "
            "Recovery analysis is planned for a future release."
        )
