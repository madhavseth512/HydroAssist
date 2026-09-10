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

Built on CurveFitCalculator: this module supplies the model, the starting point
and the interpretation rules. Fitting, confidence intervals, goodness of fit and
the shared diagnostics — bound hits, parameter correlation, AICc — come from the
template. The fitted numbers are identical to the hand-written version this
replaced; tests/fixtures/golden_fits.json pins them to 1e-9.
"""

import logging
from typing import Optional

import numpy as np
from scipy.special import expn

from src.calculator.fitting import ParameterSpec
from src.calculator.rules import (
    ValidityRule, bound_hit_rule, ci_unavailable_rule, overfitting_rule,
    parameter_correlation_rule, poor_fit_rule,
)
from src.calculator.templates import CurveFitCalculator

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

# Optimiser tolerances tightened from scipy defaults (1e-8) because T and S
# span 4-5 orders of magnitude. Default tolerances can declare convergence
# with a 20% error in T. (Fix #4)
CURVE_FIT_TOL = 1e-12

_CRUDE_GUESS_NOTE = (
    "Initial parameter guess is crude (CJ slope method not applicable — "
    "test duration may be too short for the linear regime). "
    "Verify fit quality carefully."
)


def _make_theis_model(Q: float, r: float):
    """
    Factory returning the Theis model function with Q and r as closed-over
    invariants. Q and r are fixed test parameters (Theis treats them as
    constants throughout). Signature f(t, T, S) is what the fitter calls.

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


def _late_time_slope(time_s: np.ndarray, drawdown_m: np.ndarray,
                     r: float) -> Optional[float]:
    """Semi-log slope of the last third of the record, or None when the data
    are not in the Cooper-Jacob straight-line regime or the slope is not
    positive.

    Regime pre-check: u = r²S/4Tt. Since T and S are unknown, use the ratio
    r²/4t (equivalent to T=S=1, which cancels). This is order-of-magnitude
    neutral and avoids the 1000× error that the canonical T=S=1e-4 proxy
    produces for high-transmissivity aquifers. (Fix #1 — partial)
    """
    # Use last third of data (latest time → smallest u → most linear)
    n_late = max(3, len(time_s) // 3)
    t_late = time_s[-n_late:]
    s_late = drawdown_m[-n_late:]

    u_approx = (r ** 2) / (4.0 * t_late)
    if np.all(u_approx < 0.5) and len(t_late) >= 3:
        slope, _ = np.polyfit(np.log10(t_late), s_late, 1)
        if slope > 1e-6:
            return float(slope)
    return None


def _estimate_initial_T(time_s: np.ndarray, drawdown_m: np.ndarray,
                         Q: float, r: float) -> tuple[float, list]:
    """
    Estimate T_init from late-time data using the Cooper-Jacob linearisation
    (Theis eq. 3 series truncated for small u: W(u) ≈ -0.5772 - ln(u)).

    Returns (T_init, notes_list).
    """
    slope = _late_time_slope(time_s, drawdown_m, r)
    if slope is not None:
        T_init = (2.303 * Q) / (4.0 * np.pi * slope)
        logger.debug(f"T_init from CJ slope: {T_init:.3e} m²/s")
        return T_init, []

    # Fallback: CJ slope method failed (short test or non-linear regime).
    # Setting W(u) = 1 implicitly — crude but within the right order of magnitude.
    # The bounded Trust Region Reflective fit is robust to this. (Fix #2)
    T_init = Q / (4.0 * np.pi * np.median(drawdown_m))
    logger.warning(f"T_init fallback used: {T_init:.3e} m²/s")
    return T_init, [_CRUDE_GUESS_NOTE]


class TheisCalculator(CurveFitCalculator):
    """
    Theis (1935) pumping test analysis for confined aquifers.

    Fits the Theis well function W(u) to observed time-drawdown data by bounded
    nonlinear least squares, estimating T (transmissivity) and S (storativity)
    with 95% confidence intervals.

    Aquifer conditions required (from Theis 1935):
      - Confined (artesian) aquifer, bounded above and below by aquitards
      - Homogeneous and isotropic
      - Infinite areal extent (no boundaries within radius of influence)
      - Fully penetrating pumping well
      - Constant pumping rate Q throughout the test
      - Well of infinitesimal diameter (Theis assumption 5)
    """

    method_id = "theis_1935"
    method_label = "Theis (1935)"
    fit_tolerance = CURVE_FIT_TOL
    max_evals = 10000

    def parameter_specs(self, inputs, time_s, drawdown_m):
        # Linear rather than log scale: this is the parametrisation the golden
        # baseline was recorded with, and two parameters alone are not badly
        # enough scaled against each other for the optimiser to need it.
        T_init, _ = _estimate_initial_T(time_s, drawdown_m, inputs.Q, inputs.r)
        return [
            ParameterSpec("T", "m^2/s", T_MIN, T_MAX,
                          float(np.clip(T_init, T_MIN * 10, T_MAX / 10)),
                          symbol="T", display_scale=86400.0,
                          display_unit="m^2/day", description="Transmissivity"),
            # Canonical midpoint of the confined storativity range.
            ParameterSpec("S", "-", S_MIN, S_MAX, 1e-4,
                          symbol="S", description="Storativity"),
        ]

    def model(self, inputs):
        return _make_theis_model(inputs.Q, inputs.r)

    def extra_diagnostics(self, inputs, params):
        # Largest u over the last third of the record — whether the
        # Cooper-Jacob approximation holds for this dataset. max(1, ...)
        # prevents the -0 slice bug (Fix #6).
        time_s, _ = inputs.get_series()
        n_late = max(1, len(time_s) // 3)
        u_late = (inputs.r ** 2 * params["S"]) / (4.0 * params["T"] * time_s[-n_late:])
        return {"u_max_late": float(np.max(u_late))}

    def validity_rules(self):
        return [
            _crude_initial_guess_rule(),
            ci_unavailable_rule(
                "Confidence intervals could not be computed — the covariance matrix "
                "is ill-conditioned. This often indicates aquifer heterogeneity or "
                "boundary effects (Theis 1935: 'more observations are always necessary "
                "to guard against heterogeneity of the aquifer')."
            ),
            bound_hit_rule(),
            parameter_correlation_rule(),
            overfitting_rule(),
            poor_fit_rule(),
            _storativity_rule(),
            _cooper_jacob_regime_rule(),
            _recovery_hint_rule(),
        ]


# ── Method-specific interpretation ────────────────────────────────────────────

def _crude_initial_guess_rule() -> ValidityRule:
    """The optimiser started from the fallback estimate, so it may have settled
    somewhere poor — worth saying before anything else."""
    return ValidityRule(
        id="crude_initial_guess",
        severity="warning",
        applies=lambda c: _late_time_slope(c.time_s, c.drawdown_obs, c.inputs.r) is None,
        message=lambda c: _CRUDE_GUESS_NOTE,
    )


def _storativity_rule() -> ValidityRule:
    """Storativity outside the confined range, with the paper's short-distance
    caveat: close to the pumped well S is poorly constrained while T is not."""

    def text(c):
        S, r = c.value("S"), c.inputs.r
        if S > 1e-3:
            return (
                f"S = {S:.2e} is above the typical confined aquifer range (10⁻⁵ – 10⁻³). "
                "Verify the aquifer is truly artesian. Higher S suggests unconfined "
                "or leaky conditions, for which Theis assumptions do not hold."
            )
        if r < 5.0:
            return (
                f"S = {S:.2e} is very low. With r = {r}m (short distance), "
                "storativity is poorly constrained — T is reliable but S is not. "
                "Use a more distant observation well for a reliable S estimate "
                "(Theis 1935: time-lag makes S computation unreliable at short r)."
            )
        return (
            f"S = {S:.2e} is below the typical confined range (10⁻⁵). "
            "Check for leakage or partial penetration effects."
        )

    return ValidityRule(
        id="storativity_out_of_range",
        severity="warning",
        applies=lambda c: c.value("S") > 1e-3 or c.value("S") < 1e-6,
        message=text,
        citation="Theis (1935)",
    )


def _cooper_jacob_regime_rule() -> ValidityRule:
    """Large t → small u → u < CJ_U_THRESHOLD → Cooper-Jacob valid (Theis eq. 3
    truncation). Always reported, in whichever direction it falls."""

    def text(c):
        if c.diagnostics["u_max_late"] < CJ_U_THRESHOLD:
            return (
                f"u < {CJ_U_THRESHOLD} satisfied for all late-time points "
                "(Cooper-Jacob approximation is valid for this dataset). "
                "Consider also running Cooper-Jacob (1946) for comparison."
            )
        return (
            f"u < {CJ_U_THRESHOLD} not satisfied throughout — Theis full-curve fitting "
            "is the appropriate method for this dataset."
        )

    return ValidityRule(
        id="cooper_jacob_regime",
        severity="info",
        applies=lambda c: "u_max_late" in c.diagnostics,
        message=text,
        citation="Theis (1935) eq. 3",
    )


def _recovery_hint_rule() -> ValidityRule:
    return ValidityRule(
        id="recovery_hint",
        severity="info",
        applies=lambda c: True,
        message=lambda c: (
            "If recovery data is available, Theis (1935) equation 7 provides "
            "an independent T estimate from residual drawdown vs log(t/t'). "
            "Recovery analysis is planned for a future release."
        ),
        citation="Theis (1935) eq. 7",
    )
