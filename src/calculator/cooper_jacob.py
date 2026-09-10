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

Built on StraightLineCalculator. This module supplies what is specific to
Cooper-Jacob — the u-filter deciding which points lie on the straight line, the
conversion from slope and intercept to T and S, the interval propagation and the
paper's validity checks. Validation, the regression, goodness of fit and the
result assembly belong to the template.
"""

import logging

import numpy as np

from src.calculator.fitting import ParameterSpec
from src.calculator.rules import ValidityRule, bound_hit_rule
from src.calculator.templates import Selection, StraightLineCalculator

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

# Relative change in both T and S below which the u-filter has settled.
_CONVERGENCE = 1e-3
_MAX_ITERATIONS = 10


def _parameters_from_line(Q: float, r: float, slope: float, intercept: float) -> dict:
    """Equations 8 and 9: T from the slope, S from the zero-drawdown intercept.

    t₀ is where the line crosses s = 0:  0 = slope·log₁₀(t₀) + intercept.
    """
    T = (2.303 * Q) / (4.0 * np.pi * slope)
    t0 = 10.0 ** (-intercept / slope)
    S = (2.25 * T * t0) / (r ** 2)
    return {"T": T, "S": S}


class CooperJacobCalculator(StraightLineCalculator):
    """
    Cooper-Jacob (1946) straight-line method for confined aquifer analysis.

    Fits a straight line to the s vs log₁₀(t) plot using only data points
    satisfying the u < 0.02 validity criterion (Cooper & Jacob 1946, p.3).
    The point selection is iterated to convergence: regress, recompute u from
    the new T and S, re-select, until the selection and the fit agree.

    Aquifer conditions required (from Cooper & Jacob 1946):
      - Confined aquifer (same as Theis)
      - u < 0.02 for all analysed time steps (late-time requirement)
      - Constant pumping rate Q
      - Fully penetrating pumping well

    For early-time data or short tests where u > 0.02, use Theis (1935) instead.
    """

    method_id = "cooper_jacob_1946"
    method_label = "Cooper-Jacob (1946)"
    line_failure_message = "Regression failed: non-positive slope on filtered data."

    def __init__(self, u_threshold: float = CJ_U_THRESHOLD):
        """
        Args:
            u_threshold: Maximum allowable u for the straight-line fit.
                         Default 0.02 per Cooper & Jacob (1946) p.3.
                         Can be relaxed to 0.05 per later textbook convention,
                         but this increases truncation error.
        """
        self.u_threshold = u_threshold

    def _threshold(self, inputs) -> float:
        """A per-run override in inputs.options wins over the constructor value."""
        return float(inputs.options.get("u_threshold", self.u_threshold))

    # ── Template contract ─────────────────────────────────────────────────────

    def parameter_specs(self, inputs):
        return [
            ParameterSpec("T", "m^2/s", T_MIN, T_MAX, 1e-3,
                          symbol="T", display_scale=86400.0,
                          display_unit="m^2/day",
                          description="Transmissivity, from the slope (eq. 8)"),
            ParameterSpec("S", "-", S_MIN, S_MAX, 1e-4, symbol="S",
                          description="Storativity, from the intercept (eq. 9)"),
        ]

    def line_to_parameters(self, inputs, line):
        if line.slope <= 0:
            logger.warning(f"Non-positive slope from semi-log regression: {line.slope:.4e}")
            return None
        return _parameters_from_line(inputs.Q, inputs.r, line.slope, line.intercept)

    def select_points(self, inputs, time_s, drawdown_m):
        """Converge T, S and the u-filter together.

        A fixed two-pass scheme is not self-consistent: the second-pass T and S
        differ from the first, so u recomputed on the same filtered points can
        shift and produce marginal violations in the post-fit check. Iterating
        to convergence (typically 2-3 iterations) makes the filter and the final
        fit agree.
        """
        threshold = self._threshold(inputs)
        r = inputs.r

        first = self.line_to_parameters(inputs, self.fit_line(time_s, drawdown_m))
        if first is None:
            return Selection(error=(
                "Pass-1 regression failed: slope is non-positive. "
                "Ensure drawdown increases with time and data is from the pumping phase."
            ))

        T_iter, S_iter = first["T"], first["S"]
        mask = np.ones(time_s.size, dtype=bool)
        notes, converged = [], False

        for iteration in range(_MAX_ITERATIONS):
            u = (r ** 2 * S_iter) / (4.0 * T_iter * time_s)
            valid = u < threshold
            n_valid = int(valid.sum())

            if n_valid < MIN_VALID_POINTS:
                notes.append(
                    f"Only {n_valid} point(s) satisfy u < {threshold} "
                    f"({len(time_s) - n_valid} excluded as early-time). "
                    f"Cooper-Jacob requires at least {MIN_VALID_POINTS} valid points. "
                    "Using all data as fallback — Theis (1935) is more appropriate."
                )
                mask = np.ones(time_s.size, dtype=bool)
                break

            params = self.line_to_parameters(
                inputs, self.fit_line(time_s[valid], drawdown_m[valid]))
            if params is None:
                return Selection(error=(
                    f"Regression failed on filtered data at iteration {iteration + 1}. "
                    "Check that late-time drawdown is increasing."
                ))

            mask = valid
            T_change = abs(params["T"] - T_iter) / T_iter
            S_change = abs(params["S"] - S_iter) / S_iter
            T_iter, S_iter = params["T"], params["S"]

            if T_change < _CONVERGENCE and S_change < _CONVERGENCE:
                notes.append(
                    f"Data selection converged in {iteration + 1} iteration(s): "
                    f"{n_valid} points satisfy u < {threshold}."
                )
                converged = True
                break
        else:
            notes.append(
                f"Data selection did not converge in {_MAX_ITERATIONS} iterations. "
                "Results may be sensitive to initial estimates."
            )

        n_excluded = int(time_s.size - mask.sum())
        if n_excluded > 0 and converged:
            notes.append(
                f"{n_excluded} early-time point(s) excluded (u ≥ {threshold}) — "
                "straight line fitted to late-time data only (Cooper & Jacob 1946)."
            )
        return Selection(mask=mask, notes=notes)

    def confidence_intervals(self, inputs, line, params):
        """95% intervals from the regression's standard errors.

        T = 2.303Q / (4π × slope), so its interval is asymmetric and uses the
        slope bounds swapped. S depends on both slope and intercept, so its
        interval is the range over the four corners of the (slope, intercept)
        uncertainty box — the delta method would be more compact but hides how
        strongly S depends on the extrapolated t₀.
        """
        notes = []
        unc, problem = line.uncertainty()
        if problem == "too_few_points":
            return {"T": None, "S": None}, [
                "Confidence intervals not computed — fewer than 3 valid points."]
        if problem == "no_spread":
            return {"T": None, "S": None}, [
                "Confidence intervals not computed — insufficient time range variance."]

        Q, r = inputs.Q, inputs.r
        slope_lower, slope_upper = unc.slope_interval

        if slope_lower <= 0 or slope_upper <= 0:
            notes.append(
                "T confidence interval not computed — slope CI includes non-positive values. "
                "This indicates high uncertainty in the straight-line fit."
            )
            T_ci = None
        else:
            T_ci = (_parameters_from_line(Q, r, slope_upper, line.intercept)["T"],
                    _parameters_from_line(Q, r, slope_lower, line.intercept)["T"])

        try:
            intercept_lower, intercept_upper = unc.intercept_interval
            corners = []
            for sl in (slope_lower, slope_upper):
                if sl <= 0:
                    continue
                for ic in (intercept_lower, intercept_upper):
                    S_corner = _parameters_from_line(Q, r, sl, ic)["S"]
                    # Slightly outside the fitting bounds is allowed for an interval.
                    if S_MIN <= S_corner <= S_MAX * 10:
                        corners.append(S_corner)
            if corners:
                S_ci = (min(corners), max(corners))
            else:
                S_ci = None
                notes.append(
                    "S confidence interval not computed — corner values out of physical range.")
        except Exception:
            S_ci = None

        return {"T": T_ci, "S": S_ci}, notes

    def extra_diagnostics(self, inputs, line, params):
        n_total = len(inputs.get_series()[0])
        u_fitted = (inputs.r ** 2 * params["S"]) / (4.0 * params["T"] * line.t)
        diagnostics = {
            "slope_m_per_log_cycle": line.slope,     # Δs
            "u_max_fitted": float(np.max(u_fitted)),
            "u_threshold": self._threshold(inputs),
            "n_fitted": line.n,
            "n_excluded": n_total - line.n,
        }
        if abs(line.slope) > 1e-12:
            diagnostics["t0_s"] = 10.0 ** (-line.intercept / line.slope)
        return diagnostics

    def validity_rules(self):
        return [
            bound_hit_rule(),
            _u_check_failed_rule(),
            _u_check_passed_rule(),
            _linearity_rule(),
            _storativity_rule(),
            _t0_extrapolated_rule(),
            _compare_with_theis_rule(),
            _recovery_hint_rule(),
        ]


# ── Method-specific interpretation ────────────────────────────────────────────
# The authoritative u check runs on the points actually fitted, with the clipped
# T and S that are returned — so the note agrees with the numbers shown.

def _u_violated(c) -> bool:
    return c.diagnostics["u_max_fitted"] >= c.diagnostics["u_threshold"]


def _u_check_failed_rule() -> ValidityRule:
    return ValidityRule(
        id="u_check_failed",
        severity="warning",
        applies=_u_violated,
        message=lambda c: (
            f"Post-fit u check: maximum u on fitted points = "
            f"{c.diagnostics['u_max_fitted']:.4f} "
            f"(threshold {c.diagnostics['u_threshold']}). Some points violate the "
            "validity criterion — treat T and S as approximate. Theis (1935) is more "
            "appropriate for this dataset."
        ),
        citation="Cooper & Jacob (1946) p. 3",
    )


def _u_check_passed_rule() -> ValidityRule:
    return ValidityRule(
        id="u_check_passed",
        severity="info",
        applies=lambda c: not _u_violated(c),
        message=lambda c: (
            f"Post-fit u check passed: all {c.diagnostics['n_fitted']} fitted points "
            f"satisfy u < {c.diagnostics['u_threshold']} "
            f"(max u = {c.diagnostics['u_max_fitted']:.4f}). "
            "Cooper-Jacob approximation is valid for this dataset."
        ),
        citation="Cooper & Jacob (1946) p. 3",
    )


def _linearity_rule() -> ValidityRule:
    """Stricter than the curve-fit poor-fit threshold: a straight line that is
    only moderately straight means the data are not in the Cooper-Jacob regime."""
    return ValidityRule(
        id="poor_linearity",
        severity="warning",
        applies=lambda c: c.gof.r_squared < 0.97,
        message=lambda c: (
            f"Moderate linearity (R² = {c.gof.r_squared:.3f} < 0.97) on the semi-log plot. "
            "A poor straight-line fit suggests the data may not be fully in the "
            "linear Cooper-Jacob regime — consider using Theis (1935) for a full-curve fit."
        ),
    )


def _storativity_rule() -> ValidityRule:
    def text(c):
        S = c.value("S")
        if S > 1e-3:
            return (
                f"S = {S:.2e} is above the typical confined aquifer range (10⁻⁵ – 10⁻³). "
                "Verify the aquifer is truly artesian."
            )
        return (
            f"S = {S:.2e} is below the typical confined range (10⁻⁵). "
            "Check for leakage or partial penetration."
        )

    return ValidityRule(
        id="storativity_out_of_range",
        severity="warning",
        applies=lambda c: c.value("S") > 1e-3 or c.value("S") < 1e-6,
        message=text,
    )


def _t0_extrapolated_rule() -> ValidityRule:
    return ValidityRule(
        id="t0_extrapolated",
        severity="info",
        applies=lambda c: ("t0_s" in c.diagnostics
                           and c.diagnostics["t0_s"] < float(c.time_s.min())),
        message=lambda c: (
            f"t₀ = {c.diagnostics['t0_s']:.1f}s (zero-drawdown intercept) is extrapolated "
            f"before the earliest measurement ({float(c.time_s.min()):.0f}s). This is "
            "expected when the pumping test starts after the straight-line regime is "
            "established. S is computed from this extrapolated t₀ (Cooper & Jacob 1946, "
            "eq. 9)."
        ),
        citation="Cooper & Jacob (1946) eq. 9",
    )


def _compare_with_theis_rule() -> ValidityRule:
    return ValidityRule(
        id="compare_with_theis",
        severity="info",
        applies=lambda c: True,
        message=lambda c: (
            "Theis (1935) full-curve fitting provides an independent estimate of T and S "
            "valid across all time steps. Comparing both methods is good practice."
        ),
    )


def _recovery_hint_rule() -> ValidityRule:
    return ValidityRule(
        id="recovery_hint",
        severity="info",
        applies=lambda c: True,
        message=lambda c: (
            "If recovery data is available, Theis (1935) equation 7 and the "
            "Cooper-Jacob recovery method provide an independent T estimate. "
            "Recovery analysis is planned for a future release."
        ),
    )
