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
from math import factorial
from typing import Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expn, kv

from src.calculator.base import BaseCalculator, CalculationInput, CalculationResult

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

def _stehfest_v(N: int) -> np.ndarray:
    """Precompute Stehfest V coefficients for N-term inversion (N must be even)."""
    M = N // 2
    V = np.zeros(N)
    for i in range(1, N + 1):
        s = 0.0
        for k in range(int(np.floor((i + 1) / 2)), min(i, M) + 1):
            s += (k ** M * factorial(2 * k) /
                  (factorial(M - k) * factorial(k) * factorial(k - 1) *
                   factorial(i - k) * factorial(2 * k - i)))
        V[i - 1] = ((-1) ** (M + i)) * s
    return V


_V12 = _stehfest_v(_STEHFEST_N)   # precomputed at import time


def _laplace_sw(p: float, Q: float, T: float, S: float,
                r_w: float, r_c: float) -> float:
    """Laplace transform of s_w at r = r_w — paper Eq. 10.

    s̄_w(p) = Q · K₀(q·r_w) / {π · p · [r_c²·p·K₀(q·r_w) + 2·r_w·T·q·K₁(q·r_w)]}

    where q = √(p·S/T).

    This is the exact analytical Laplace-domain solution.  K₀ and K₁ are
    scipy.special.kv(0, ·) and kv(1, ·) — modified Bessel functions of the
    second kind, non-oscillatory and numerically stable.
    """
    q = np.sqrt(p * S / T)
    qrw = q * r_w
    if qrw < 1e-300:
        return 0.0
    K0 = float(kv(0, qrw))
    K1 = float(kv(1, qrw))
    denom = np.pi * p * (r_c ** 2 * p * K0 + 2.0 * r_w * T * q * K1)
    if abs(denom) < 1e-300:
        return 0.0
    return Q * K0 / denom


def _stehfest_sw(t: float, Q: float, T: float, S: float,
                 r_w: float, r_c: float) -> float:
    """Invert s̄_w(p) → s_w(t) using the Stehfest (1970) algorithm.

    f(t) ≈ (ln2/t) × Σᵢ₌₁ᴺ Vᵢ · F̄(i · ln2/t)
    """
    ln2_t = np.log(2.0) / t
    s_w = 0.0
    for i in range(1, _STEHFEST_N + 1):
        p_i = i * ln2_t
        s_w += _V12[i - 1] * _laplace_sw(p_i, Q, T, S, r_w, r_c)
    return ln2_t * s_w


def pc_drawdown(t_arr: np.ndarray, T: float, S: float,
                Q: float, r_w: float, r_c: float) -> np.ndarray:
    """Drawdown at the pumped well face via Stehfest Laplace inversion.

    Implements paper Eq. 12 by inverting the Laplace-domain formula (Eq. 10)
    rather than integrating the oscillatory Eq. 13 directly.

    Module-level so report.py and plots.py can import it for PDF plots and
    interactive matching without re-importing the whole calculator class.
    """
    out = np.empty(len(t_arr))
    for i, t in enumerate(t_arr):
        out[i] = _stehfest_sw(t, Q, T, S, r_w, r_c)
    return out


# ── Calculator class ──────────────────────────────────────────────────────────

class PapadopulosCooperCalculator(BaseCalculator):
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
    (Papadopulos & Cooper 1967, p. 244).
    """

    def calculate(self, inputs: CalculationInput) -> CalculationResult:
        error = self._validate(inputs)
        if error:
            return CalculationResult(
                method="Papadopulos-Cooper (1967)", T=0.0, S=0.0,
                success=False, error_message=error,
            )

        time_s, drawdown_m = inputs.get_series()
        Q, r_w, r_c = inputs.Q, inputs.r_w, inputs.r_c
        notes = []

        # ── Initial parameter estimates ───────────────────────────────────────
        T_init = self._estimate_T_init(time_s, drawdown_m, Q, r_w)
        T_init = float(np.clip(T_init, T_MIN * 10, T_MAX / 10))
        S_init = 1e-4

        # ── Curve fit: free parameters are T and S; Q, r_w, r_c are fixed ────
        model = lambda t, T, S: pc_drawdown(t, T, S, Q, r_w, r_c)

        try:
            popt, pcov = curve_fit(
                model,
                time_s,
                drawdown_m,
                p0=[T_init, S_init],
                bounds=([T_MIN, S_MIN], [T_MAX, S_MAX]),
                maxfev=5000,
                method="trf",
                ftol=CURVE_FIT_TOL,
                xtol=CURVE_FIT_TOL,
                gtol=CURVE_FIT_TOL,
            )
        except (RuntimeError, ValueError) as e:
            return CalculationResult(
                method="Papadopulos-Cooper (1967)", T=0.0, S=0.0,
                success=False,
                error_message=(
                    f"Curve fitting failed: {e}. "
                    "Ensure data are from the pumped well and span the curved "
                    "portion of the type curve (not only the early-time straight "
                    "portion — paper p. 244)."
                ),
            )

        T_fit = float(np.clip(popt[0], T_MIN, T_MAX))
        S_fit = float(np.clip(popt[1], S_MIN, S_MAX))
        alpha = (r_w ** 2 * S_fit) / (r_c ** 2)

        # ── Confidence intervals ───────────────────────────────────────────────
        T_ci, S_ci = self._compute_ci(pcov, T_fit, S_fit, notes)

        # ── Goodness-of-fit ───────────────────────────────────────────────────
        s_fitted = pc_drawdown(time_s, T_fit, S_fit, Q, r_w, r_c)
        residuals = drawdown_m - s_fitted
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((drawdown_m - np.mean(drawdown_m)) ** 2))
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # ── Validity notes ────────────────────────────────────────────────────
        self._add_validity_notes(
            T_fit, S_fit, alpha, r_squared, time_s, Q, r_w, r_c, notes
        )

        logger.info(
            f"PC fit: T={T_fit:.3e} m²/s ({T_fit*86400:.1f} m²/day), "
            f"S={S_fit:.3e}, α={alpha:.2e}, R²={r_squared:.4f}, RMSE={rmse:.4f}m"
        )

        return CalculationResult(
            method="Papadopulos-Cooper (1967)",
            T=T_fit,
            S=S_fit,
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
        if inputs.Q <= 0:
            return f"Q must be positive (got {inputs.Q} m³/s)."
        if inputs.r_w <= 0:
            return f"r_w must be positive (got {inputs.r_w} m)."
        if inputs.r_c <= 0:
            return f"r_c must be positive (got {inputs.r_c} m)."
        t, s = inputs.get_series()
        if len(t) < 10:
            return f"Insufficient data: {len(t)} points (minimum 10 required)."
        if np.any(s < 0):
            return "Negative drawdown values detected — remove recovery phase data."
        if np.any(t <= 0):
            return "Non-positive time values detected."
        return None

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

    def _compute_ci(self, pcov: np.ndarray, T_fit: float, S_fit: float,
                    notes: list) -> tuple:
        diag = np.diag(pcov)
        if np.any(np.isinf(diag)) or np.any(diag < 0):
            notes.append(
                "Confidence intervals could not be computed — covariance matrix "
                "is ill-conditioned. This often means the data do not uniquely "
                "constrain both T and S (paper p. 244: type curves differ only "
                "slightly when α changes by an order of magnitude)."
            )
            return None, None
        perr = np.sqrt(diag)
        T_ci = (T_fit - 1.96 * perr[0], T_fit + 1.96 * perr[0])
        S_ci = (S_fit - 1.96 * perr[1], S_fit + 1.96 * perr[1])
        return T_ci, S_ci

    def _add_validity_notes(
        self,
        T: float, S: float, alpha: float, r_sq: float,
        time_s: np.ndarray, Q: float,
        r_w: float, r_c: float,
        notes: list,
    ):
        # Paper p.244 mandatory S warning
        notes.append(
            f"S = {S:.2e} has QUESTIONABLE RELIABILITY — paper p. 244: "
            "'the determined value of S will change by an order of magnitude "
            "when the data plot is moved from one type curve to another.' "
            "T is the reliable output from this method."
        )

        # α value and meaning
        notes.append(
            f"α = r_w²S/r_c² = {alpha:.3e}. "
            "Large α (> 0.1): wellbore storage dominates at early time. "
            "Small α (< 10⁻⁴): converges quickly to Theis."
        )

        # Fraction of data in each regime
        u_w_arr = (r_w ** 2 * S) / (4.0 * T * time_s)
        n_theis = int(np.sum(u_w_arr / alpha < 1e-3))
        n_ws = int(np.sum(u_w_arr / alpha > 1.0))
        notes.append(
            f"Regime breakdown: {n_theis}/{len(time_s)} points in Theis regime "
            f"(u_w/α < 10⁻³), {n_ws}/{len(time_s)} in wellbore-storage regime "
            f"(u_w/α > 1). PC fitting requires points on the curved transition."
        )

        # Late-time convergence threshold (paper p.242)
        t_conv = 2.5e3 * r_c ** 2 / T
        notes.append(
            f"Convergence to Theis after t ≈ {t_conv:.0f} s "
            f"(paper criterion: t > 2.5×10³ × r_c²/T = 2500 × {r_c**2:.4f} / {T:.3e})."
        )

        # Goodness-of-fit
        if r_sq < 0.95:
            notes.append(
                f"Poor fit (R² = {r_sq:.3f}). If data fall entirely on the "
                "early-time straight portion of the type curve (s_w ≈ Qt/πr_c²), "
                "aquifer properties cannot be recovered — paper p. 244."
            )

        # Early-time straight-line check (wellbore storage dominated)
        t_ws_limit = Q * time_s / (np.pi * r_c ** 2)
        s_obs = None  # Not passed here; check via u_w
        if n_ws > len(time_s) * 0.8:
            notes.append(
                "WARNING: Most data appear to be in the wellbore-storage-dominated "
                "regime. The type curve method cannot reliably estimate aquifer "
                "properties from data lying entirely on the early-time straight "
                "portion (paper p. 244)."
            )

        # Comparison recommendation
        notes.append(
            "Run Theis (1935) on the same data: at late time both methods should "
            "agree on T. If they diverge significantly, check for boundary effects "
            "or aquifer heterogeneity."
        )
