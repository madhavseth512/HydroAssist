"""Papadopulos-Cooper (1967) large-diameter well calculator.

Reference: Papadopulos, I.S. & Cooper, H.H. (1967). Drawdown in a well of large diameter.
Water Resources Research, 3(1), 241-244. doi:10.1029/WR003i001p00241

Key equations from the paper:

  Equation 12 — Drawdown at the pumped well:
      s_w = Q / (4πT) × F(u_w, α)

  Equation 13 — Well function:
      F(u_w, α) = (32α² / π²) × ∫₀^∞  (1 − e^{−β²/4u_w}) / (β³ × Δ(β))  dβ

  Δ(β) denominator (from Eq. 11):
      Δ(β) = [β·J₀(β) − 2α·J₁(β)]² + [β·Y₀(β) − 2α·Y₁(β)]²

  Parameters (paper p. 242):
      α   = r_w² × S / r_c²          (dimensionless storage parameter)
      u_w = r_w² × S / (4 × T × t)   (dimensionless time at well face)

  Late-time convergence to Theis (paper p. 242):
      When u_w/α < 10⁻³  (i.e., t > 2.5×10³ × r_c²/T):
          F(u_w, α) ≈ W(u_w) = expn(1, u_w)

  Critical limitation (paper p. 244):
      "a determination of S by this method has questionable reliability.
      Whereas the determined value of S will change by an order of magnitude
      when the data plot is moved from one type curve to another, that of T
      will change only slightly."
      → T is the reliable output.  S has order-of-magnitude uncertainty.

  Data requirement (paper p. 244):
      "For an analysis to be possible, most of the data should fall on the
      curved part of the type curves."  If all data lie on the early-time
      straight portion (wellbore storage dominated), the method cannot recover
      aquifer properties.
"""

import logging

import numpy as np

from src.calculator.fitting import ParameterSpec
from src.calculator.laplace import stehfest_coefficients, stehfest_invert_array
from src.calculator.laplace_solutions import (
    papadopulos_cooper_transform, q_theis, with_wellbore_storage,
)
from src.calculator.rules import (
    ValidityRule, bound_hit_rule, ci_unavailable_rule, overfitting_rule,
)
from src.calculator.templates import CurveFitCalculator

logger = logging.getLogger(__name__)

# ── Physical bounds (same philosophy as Theis/CJ) ────────────────────────────
T_MIN, T_MAX = 1e-9, 1.0      # [m²/s]
S_MIN, S_MAX = 1e-9, 1e-2     # [dimensionless]

CURVE_FIT_TOL = 1e-10

# Stehfest N=12 is industry standard for smooth Laplace transforms
_STEHFEST_N = 12


# ── Stehfest (1970) numerical Laplace inversion ───────────────────────────────
# We invert the Laplace-domain solution (paper Eq. 10 at r = r_w) instead of
# numerically integrating the oscillatory Eq. 13 integral.  The Laplace-domain
# formula involves only K₀ and K₁ (modified Bessel functions), which are smooth
# and well-conditioned — no oscillation, no convergence issues.
#
# The machinery itself now lives in laplace.py and laplace_solutions.py, shared
# with the leaky, unconfined and fractured methods that use the same kernel with
# a different characteristic root. The four names below are kept as thin wrappers
# because they are part of this module's tested surface.


def _stehfest_v(N: int) -> np.ndarray:
    """Stehfest V coefficients for an N-term inversion (N must be even)."""
    return np.asarray(stehfest_coefficients(N), dtype=float)


_V12 = _stehfest_v(_STEHFEST_N)   # precomputed at import time


def _laplace_sw(p, Q: float, T: float, S: float,
                r_w: float, r_c: float):
    """Laplace transform of s_w at r = r_w — paper Eq. 10.

    s̄_w(p) = Q · K₀(q·r_w) / {π · p · [r_c²·p·K₀(q·r_w) + 2·r_w·T·q·K₁(q·r_w)]}

    where q = √(p·S/T).

    Accepts a scalar or an array of p. K₀ and K₁ appear only in a ratio at the
    same argument, so the exponentially scaled forms used underneath substitute
    exactly while staying finite where the plain functions both underflow.
    """
    out = with_wellbore_storage(p, Q, T, r_w, r_w, r_c, q_theis(p, T, S))
    return float(out) if np.ndim(p) == 0 else out


def _stehfest_sw(t: float, Q: float, T: float, S: float,
                 r_w: float, r_c: float) -> float:
    """Invert s̄_w(p) → s_w(t) at a single time.

    f(t) ≈ (ln2/t) × Σᵢ₌₁ᴺ Vᵢ · F̄(i · ln2/t)
    """
    return float(stehfest_invert_array(
        papadopulos_cooper_transform(Q, T, S, r_w, r_c), float(t), _STEHFEST_N
    ))


def pc_drawdown(t_arr: np.ndarray, T: float, S: float,
                Q: float, r_w: float, r_c: float) -> np.ndarray:
    """Drawdown at the pumped well face via Stehfest Laplace inversion.

    Implements paper Eq. 12 by inverting the Laplace-domain formula (Eq. 10)
    rather than integrating the oscillatory Eq. 13 directly.

    The transform is evaluated once over the whole (n_times × N) grid rather than
    in a Python loop per time point. That is what makes a multi-parameter fit on
    this kernel tractable — the loop cost roughly N calls per time point, and a
    four- to six-parameter fit needs on the order of 10^5 evaluations.

    Module-level so report.py and plots.py can import it for PDF plots and
    interactive matching without re-importing the whole calculator class.
    """
    return stehfest_invert_array(
        papadopulos_cooper_transform(Q, T, S, r_w, r_c),
        np.asarray(t_arr, dtype=float),
        _STEHFEST_N,
    )


# ── Calculator class ──────────────────────────────────────────────────────────

_S_UNRELIABLE = (
    "Papadopulos & Cooper (1967) p. 244: the determined value of S changes by an "
    "order of magnitude when the data are moved from one type curve to another, "
    "while T changes only slightly."
)


class PapadopulosCooperCalculator(CurveFitCalculator):
    """
    Papadopulos-Cooper (1967) large-diameter well analysis.

    Fits the well function F(u_w, α) to observed time-drawdown data recorded
    at the pumped well itself.  The method accounts for wellbore storage
    (water draining from within the well at early time), which the Theis
    line-source assumption ignores.

    Aquifer conditions required (paper §Analysis):
      - Confined (artesian) homogeneous isotropic aquifer
      - Pumped well fully penetrating
      - Constant discharge Q
      - Data from the pumped well (not an observation well)

    T is reliably determined; S has order-of-magnitude uncertainty
    (Papadopulos & Cooper 1967, p. 244), and is reported flagged as such.

    Built on CurveFitCalculator; the fitted numbers are identical to the
    hand-written version this replaced (tests/fixtures/golden_fits.json).
    """

    method_id = "papadopulos_cooper_1967"
    method_label = "Papadopulos-Cooper (1967)"
    requires = ("well.r_w", "well.r_c")
    fit_tolerance = CURVE_FIT_TOL
    max_evals = 5000

    def parameter_specs(self, inputs, time_s, drawdown_m):
        T_init = self._estimate_T_init(time_s, drawdown_m, inputs.Q, inputs.r_w)
        return [
            ParameterSpec("T", "m^2/s", T_MIN, T_MAX,
                          float(np.clip(T_init, T_MIN * 10, T_MAX / 10)),
                          symbol="T", display_scale=86400.0,
                          display_unit="m^2/day", description="Transmissivity"),
            ParameterSpec("S", "-", S_MIN, S_MAX, 1e-4,
                          symbol="S", description="Storativity"),
        ]

    def model(self, inputs):
        # Free parameters are T and S; Q, r_w and r_c are fixed by the test.
        Q, r_w, r_c = inputs.Q, inputs.r_w, inputs.r_c

        def papadopulos_cooper(t, T, S):
            return pc_drawdown(t, T, S, Q, r_w, r_c)

        return papadopulos_cooper

    def parameter_metadata(self):
        return {"S": {"reliable": False, "reliability_note": _S_UNRELIABLE}}

    def extra_diagnostics(self, inputs, params):
        time_s, _ = inputs.get_series()
        T, S = params["T"], params["S"]
        r_w, r_c = inputs.r_w, inputs.r_c
        alpha = (r_w ** 2 * S) / (r_c ** 2)
        u_w = (r_w ** 2 * S) / (4.0 * T * time_s)
        return {
            "alpha": alpha,
            # Paper p. 242: F(u_w, α) ≈ W(u_w) once u_w/α < 10⁻³.
            "t_convergence_s": 2.5e3 * r_c ** 2 / T,
            "n_theis_regime": int(np.sum(u_w / alpha < 1e-3)),
            "n_wellbore_storage_regime": int(np.sum(u_w / alpha > 1.0)),
        }

    def validity_rules(self):
        # The shared parameter-correlation rule is deliberately left out. T and S
        # are strongly correlated here by construction — on the bundled sample
        # r = -0.99 — yet T is still pinned to about ±0.2%. That rule would tell
        # the user every estimate is unreliable; the paper's own finding, which
        # is the accurate one, is that S is unreliable and T is not. S is flagged
        # as such through parameter_metadata and the first method rule below.
        return [
            ci_unavailable_rule(
                "Confidence intervals could not be computed — covariance matrix "
                "is ill-conditioned. This often means the data do not uniquely "
                "constrain both T and S (paper p. 244: type curves differ only "
                "slightly when α changes by an order of magnitude)."
            ),
            bound_hit_rule(),
            overfitting_rule(),
            _storativity_reliability_rule(),
            _alpha_rule(),
            _regime_rule(),
            _convergence_rule(),
            _poor_fit_rule(),
            _wellbore_dominated_rule(),
            _compare_with_theis_rule(),
        ]

    def _estimate_T_init(self, time_s: np.ndarray, drawdown_m: np.ndarray,
                         Q: float, r_w: float) -> float:
        """Crude T estimate from late-time CJ slope (same logic as Theis)."""
        n = max(3, len(time_s) // 3)
        t_late = time_s[-n:]
        s_late = drawdown_m[-n:]
        slope, _ = np.polyfit(np.log10(t_late), s_late, 1)
        if slope > 1e-6:
            return (2.303 * Q) / (4.0 * np.pi * slope)
        return Q / (4.0 * np.pi * np.median(drawdown_m))


# ── Method-specific interpretation ────────────────────────────────────────────

def _storativity_reliability_rule() -> ValidityRule:
    """Paper p. 244 — mandatory, whatever the fit looks like."""
    return ValidityRule(
        id="storativity_unreliable",
        severity="warning",
        applies=lambda c: True,
        message=lambda c: (
            f"S = {c.value('S'):.2e} has QUESTIONABLE RELIABILITY — paper p. 244: "
            "'the determined value of S will change by an order of magnitude "
            "when the data plot is moved from one type curve to another.' "
            "T is the reliable output from this method."
        ),
        citation="Papadopulos & Cooper (1967) p. 244",
    )


def _alpha_rule() -> ValidityRule:
    return ValidityRule(
        id="wellbore_storage_parameter",
        severity="info",
        applies=lambda c: "alpha" in c.diagnostics,
        message=lambda c: (
            f"α = r_w²S/r_c² = {c.diagnostics['alpha']:.3e}. "
            "Large α (> 0.1): wellbore storage dominates at early time. "
            "Small α (< 10⁻⁴): converges quickly to Theis."
        ),
    )


def _regime_rule() -> ValidityRule:
    return ValidityRule(
        id="regime_breakdown",
        severity="info",
        applies=lambda c: "n_theis_regime" in c.diagnostics,
        message=lambda c: (
            f"Regime breakdown: {c.diagnostics['n_theis_regime']}/{len(c.time_s)} "
            "points in Theis regime (u_w/α < 10⁻³), "
            f"{c.diagnostics['n_wellbore_storage_regime']}/{len(c.time_s)} in "
            "wellbore-storage regime (u_w/α > 1). PC fitting requires points on "
            "the curved transition."
        ),
    )


def _convergence_rule() -> ValidityRule:
    def text(c):
        r_c, T = c.inputs.r_c, c.value("T")
        return (
            f"Convergence to Theis after t ≈ {c.diagnostics['t_convergence_s']:.0f} s "
            f"(paper criterion: t > 2.5×10³ × r_c²/T = 2500 × {r_c**2:.4f} / {T:.3e})."
        )

    return ValidityRule(
        id="theis_convergence_time",
        severity="info",
        applies=lambda c: "t_convergence_s" in c.diagnostics,
        message=text,
        citation="Papadopulos & Cooper (1967) p. 242",
    )


def _poor_fit_rule() -> ValidityRule:
    """The paper-specific reading of a poor fit, rather than the generic one."""
    return ValidityRule(
        id="poor_fit",
        severity="warning",
        applies=lambda c: c.gof.r_squared < 0.95,
        message=lambda c: (
            f"Poor fit (R² = {c.gof.r_squared:.3f}). If data fall entirely on the "
            "early-time straight portion of the type curve (s_w ≈ Qt/πr_c²), "
            "aquifer properties cannot be recovered — paper p. 244."
        ),
        citation="Papadopulos & Cooper (1967) p. 244",
    )


def _wellbore_dominated_rule() -> ValidityRule:
    return ValidityRule(
        id="wellbore_storage_dominated",
        severity="warning",
        applies=lambda c: (c.diagnostics.get("n_wellbore_storage_regime", 0)
                           > len(c.time_s) * 0.8),
        message=lambda c: (
            "WARNING: Most data appear to be in the wellbore-storage-dominated "
            "regime. The type curve method cannot reliably estimate aquifer "
            "properties from data lying entirely on the early-time straight "
            "portion (paper p. 244)."
        ),
        citation="Papadopulos & Cooper (1967) p. 244",
    )


def _compare_with_theis_rule() -> ValidityRule:
    return ValidityRule(
        id="compare_with_theis",
        severity="info",
        applies=lambda c: True,
        message=lambda c: (
            "Run Theis (1935) on the same data: at late time both methods should "
            "agree on T. If they diverge significantly, check for boundary effects "
            "or aquifer heterogeneity."
        ),
    )
