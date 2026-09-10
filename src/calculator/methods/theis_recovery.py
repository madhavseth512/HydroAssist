"""Theis (1935) recovery method.

Reference: Theis, C.V. (1935). The relation between the lowering of the
piezometric surface and the rate and duration of discharge of a well using
ground-water storage. Transactions of the American Geophysical Union, 16,
519-524 — equation 7.

The first method for a test type other than constant-rate pumping. When the pump
is switched off after running for t_p, the water level recovers. By
superposition, recovery is the original drawdown plus that of an imaginary
recharge well of the same rate starting at t_p, so the residual drawdown is

    s' = Q/(4πT) · [W(u) − W(u')]       u = r²S/4Tt,   u' = r²S'/4Tt'

with t the time since pumping started and t' = t − t_p the time since it
stopped. Once u and u' are both small, each W takes its logarithmic form and

    s' = (2.303Q / 4πT) · log₁₀( (t/t') · (S'/S) )          paper eq. 7

so s' against log₁₀(t/t') is a straight line. Its slope gives T exactly as
Cooper-Jacob's does. Its intercept gives only the RATIO S/S' — the line meets
s' = 0 at t/t' = S/S'. Storativity itself cannot be recovered from recovery
data, and this method does not report one.

Why run it when the pumping phase already gives T: the discharge during
recovery is exactly zero, so the record is free of the pump-rate fluctuations
that disturb a real drawdown curve. Agreement between the two T values is the
check a reviewer will ask for.

Built on StraightLineCalculator. Point selection bounds u' < 0.02 using the
largest storativity a confined aquifer plausibly has, because recovery data
cannot supply S; an S estimate from the pumping phase, passed as
options["S_estimate"], tightens the bound and lets more of the record be used.
"""

import logging
from typing import Optional

import numpy as np
from scipy.special import expn

from src.calculator.fitting import ParameterSpec
from src.calculator.rules import ValidityRule, bound_hit_rule
from src.calculator.templates import Selection, StraightLineCalculator

logger = logging.getLogger(__name__)

T_MIN, T_MAX = 1e-9, 1.0              # [m²/s]

# S/S' — one when storage is the same during pumping and recovery.
RATIO_MIN, RATIO_MAX = 1e-6, 1e6

# The same series-truncation criterion as Cooper-Jacob (1946), applied to u'.
U_THRESHOLD = 0.02

# The largest storativity of a confined aquifer (Theis 1935: 10⁻⁵ to 10⁻³).
# Used to bound u' when no estimate of S is supplied — conservative, so a
# reading that passes is valid for any confined S.
S_CONFINED_MAX = 1e-3

MIN_VALID_POINTS = 4

# S/S' further than this factor from one is worth explaining to the user.
RATIO_TOLERANCE = 2.0

SINCE_STOP = "since_stop"
SINCE_START = "since_start"
CLOCKS = (SINCE_STOP, SINCE_START)

_CONVERGENCE = 1e-3
_MAX_ITERATIONS = 10


def pumping_duration(inputs) -> Optional[float]:
    """t_p: when the schedule's rate first drops to zero after pumping, or None."""
    schedule = getattr(inputs, "schedule", None)
    if schedule is None:
        return None
    for step in schedule.steps[1:]:
        if step.Q == 0:
            return float(step.t_start_s)
    return None


def residual_drawdown(t_since_stop, T: float, S: float, Q: float, r: float,
                      t_p: float, S_prime: Optional[float] = None) -> np.ndarray:
    """Exact Theis residual drawdown by superposition, without the logarithmic
    approximation — for tests, synthetic data and forward prediction."""
    t_prime = np.asarray(t_since_stop, dtype=float)
    S_prime = S if S_prime is None else S_prime
    u = r ** 2 * S / (4.0 * T * (t_p + t_prime))
    u_prime = r ** 2 * S_prime / (4.0 * T * t_prime)
    return Q / (4.0 * np.pi * T) * (expn(1, u) - expn(1, u_prime))


def plot_axis(result):
    """t/t' for each reading of a recovery result, and a mask of the readings
    the line was fitted to. Shared by the on-screen plots and the PDF."""
    t_prime = np.asarray(result.time_s, dtype=float)
    t_p = float(result.diagnostics.get("pumping_duration_s", 0.0))
    fitted = t_prime >= float(result.diagnostics.get("t_prime_min_fitted", 0.0))
    return (t_p + t_prime) / t_prime, fitted


def _parameters_from_line(Q: float, slope: float, intercept: float) -> dict:
    """Slope → T (eq. 7, as Cooper-Jacob's eq. 8); intercept → S/S'."""
    return {
        "T": (2.303 * Q) / (4.0 * np.pi * slope),
        "S_ratio": 10.0 ** (-intercept / slope),
    }


class TheisRecoveryCalculator(StraightLineCalculator):
    """Theis (1935) recovery: T from residual drawdown against log₁₀(t/t')."""

    method_id = "theis_recovery_1935"
    method_label = "Theis Recovery (1935)"
    line_failure_message = (
        "Regression failed: residual drawdown does not fall as t/t′ decreases. "
        "Check that the readings are from the recovery phase and that the "
        "pumping duration is right."
    )

    def _S_bound(self, inputs) -> float:
        return float(inputs.options.get("S_estimate", S_CONFINED_MAX))

    # ── Template contract ─────────────────────────────────────────────────────

    def validate_inputs(self, inputs):
        # Not the shared checks: residual drawdown may legitimately dip below
        # zero once the water level is back at static, which they would reject.
        if inputs.Q is None or inputs.Q <= 0:
            return ("Pumping rate Q must be positive — enter the rate the well "
                    "pumped at before it was switched off, in m³/s.")
        t_p = pumping_duration(inputs)
        if t_p is None or t_p <= 0:
            return ("Theis recovery needs the pumping duration t_p — how long the "
                    "well pumped before it was switched off. Enter it in Step 2.")
        clock = inputs.options.get("recovery_clock", SINCE_STOP)
        if clock not in CLOCKS:
            return (f"Unknown recovery clock {clock!r}; expected one of "
                    f"{', '.join(CLOCKS)}.")
        t, _ = inputs.get_series()
        if len(t) < self.min_points:
            return (f"Insufficient data: {len(t)} points (minimum "
                    f"{self.min_points} required).")
        if np.any(t < 0):
            return "Negative time values detected — check the time column."
        return None

    def prepare_series(self, inputs, time_s, drawdown_m):
        """Convert the clock to t', time since the pump stopped, and keep only
        the recovery readings."""
        t_p = pumping_duration(inputs)
        notes = []
        if inputs.options.get("recovery_clock", SINCE_STOP) == SINCE_START:
            t_prime = np.asarray(time_s, dtype=float) - t_p
            n_pumping = int(np.sum(t_prime <= 0))
            if n_pumping:
                notes.append(
                    f"{n_pumping} reading(s) at or before the pump was switched off "
                    f"(t ≤ t_p = {t_p:.0f} s) belong to the pumping phase and were "
                    "left out."
                )
        else:
            t_prime = np.asarray(time_s, dtype=float)
            n_zero = int(np.sum(t_prime <= 0))
            if n_zero:
                notes.append(
                    f"{n_zero} reading(s) at t′ = 0 left out — t/t′ is undefined at "
                    "the moment the pump stops."
                )

        keep = t_prime > 0
        if keep.sum() < self.min_points:
            raise ValueError(
                f"Only {int(keep.sum())} recovery readings remain after the pump was "
                f"switched off (minimum {self.min_points}). Check the pumping "
                "duration and whether the time column counts from the start or the "
                "end of pumping."
            )
        return t_prime[keep], np.asarray(drawdown_m, dtype=float)[keep], notes

    def abscissa(self, inputs, time_s):
        t_p = pumping_duration(inputs)
        return np.log10((t_p + time_s) / time_s)

    def parameter_specs(self, inputs):
        return [
            ParameterSpec("T", "m^2/s", T_MIN, T_MAX, 1e-3,
                          symbol="T", display_scale=86400.0,
                          display_unit="m^2/day",
                          description="Transmissivity, from the slope"),
            ParameterSpec("S_ratio", "-", RATIO_MIN, RATIO_MAX, 1.0,
                          symbol="S/S'",
                          description=("Storativity during pumping over that "
                                       "during recovery, from the intercept")),
        ]

    def line_to_parameters(self, inputs, line):
        if line.slope <= 0:
            logger.warning(f"Non-positive recovery slope: {line.slope:.4e}")
            return None
        return _parameters_from_line(inputs.Q, line.slope, line.intercept)

    def select_points(self, inputs, time_s, drawdown_m):
        """Late recovery only: u' = r²S/4Tt' < 0.02, iterated with T to
        convergence, falling back to every reading if too few qualify."""
        r, S_bound = inputs.r, self._S_bound(inputs)

        first = self.line_to_parameters(inputs, self.fit_line(inputs, time_s, drawdown_m))
        if first is None:
            return Selection(error=self.line_failure_message)

        T_iter = first["T"]
        mask = np.ones(time_s.size, dtype=bool)
        notes, converged = [], False

        for _ in range(_MAX_ITERATIONS):
            u_prime = r ** 2 * S_bound / (4.0 * T_iter * time_s)
            valid = u_prime < U_THRESHOLD
            n_valid = int(valid.sum())

            if n_valid < MIN_VALID_POINTS:
                notes.append(
                    f"Only {n_valid} reading(s) are late enough to satisfy "
                    f"u′ < {U_THRESHOLD} (with S = {S_bound:.0e}), so the whole "
                    "recovery record was used. Early recovery may bend away from "
                    "the straight line — supply an S estimate from the pumping "
                    "phase, or record recovery for longer."
                )
                mask = np.ones(time_s.size, dtype=bool)
                break

            params = self.line_to_parameters(
                inputs, self.fit_line(inputs, time_s[valid], drawdown_m[valid]))
            if params is None:
                return Selection(error=self.line_failure_message)

            mask = valid
            change = abs(params["T"] - T_iter) / T_iter
            T_iter = params["T"]
            if change < _CONVERGENCE:
                converged = True
                break
        else:
            notes.append(
                f"Point selection did not settle in {_MAX_ITERATIONS} iterations; "
                "treat T as approximate."
            )

        n_excluded = int(time_s.size - mask.sum())
        if converged and n_excluded:
            basis = ("the S estimate supplied" if "S_estimate" in inputs.options
                     else f"S = {S_bound:.0e}, the largest storativity of a confined "
                          "aquifer, because recovery data cannot determine S")
            notes.append(
                f"{n_excluded} early-recovery reading(s) left out where "
                f"u′ = r²S/4Tt′ ≥ {U_THRESHOLD}, bounded using {basis}. The line "
                f"is fitted to the {int(mask.sum())} latest readings."
            )
        return Selection(mask=mask, notes=notes)

    def confidence_intervals(self, inputs, line, params):
        unc, problem = line.uncertainty()
        if problem:
            return {"T": None, "S_ratio": None}, [
                "Confidence intervals not computed — too few readings, or no "
                "spread in t/t′."]

        lo, hi = unc.slope_interval
        if lo <= 0:
            return {"T": None, "S_ratio": None}, [
                "Confidence intervals not computed — the slope's interval includes "
                "zero, so the line is too uncertain to bound T."]

        Q = inputs.Q
        T_ci = (_parameters_from_line(Q, hi, line.intercept)["T"],
                _parameters_from_line(Q, lo, line.intercept)["T"])
        corners = [_parameters_from_line(Q, sl, ic)["S_ratio"]
                   for sl in (lo, hi) for ic in unc.intercept_interval]
        ratio_ci = ((min(corners), max(corners))
                    if np.all(np.isfinite(corners)) else None)
        return {"T": T_ci, "S_ratio": ratio_ci}, []

    def extra_diagnostics(self, inputs, line, params):
        n_recovery = len(self.prepare_series(inputs, *inputs.get_series())[0])
        u_fitted = inputs.r ** 2 * self._S_bound(inputs) / (4.0 * params["T"] * line.t)
        return {
            "pumping_duration_s": pumping_duration(inputs),
            "slope_m_per_log_cycle": line.slope,
            "u_prime_max_fitted": float(np.max(u_fitted)),
            "S_bound": self._S_bound(inputs),
            "t_prime_min_fitted": float(line.t.min()),
            "n_fitted": line.n,
            "n_excluded": n_recovery - line.n,
        }

    def validity_rules(self):
        return [
            bound_hit_rule(),
            _linearity_rule(),
            _storage_ratio_far_rule(),
            _storage_ratio_near_rule(),
            _s_not_determined_rule(),
            _compare_with_pumping_rule(),
        ]


# ── Method-specific interpretation ────────────────────────────────────────────

def _ratio_is_far(c) -> bool:
    ratio = c.value("S_ratio", 1.0)
    return not (1.0 / RATIO_TOLERANCE <= ratio <= RATIO_TOLERANCE)


def _linearity_rule() -> ValidityRule:
    return ValidityRule(
        id="poor_linearity",
        severity="warning",
        applies=lambda c: c.gof.r_squared < 0.97,
        message=lambda c: (
            f"Moderate linearity (R² = {c.gof.r_squared:.3f} < 0.97) on the recovery "
            "plot. The readings may not yet follow the late-recovery straight line "
            "of equation 7 — record recovery for longer, or check the pumping "
            "duration."
        ),
        citation="Theis (1935) eq. 7",
    )


def _storage_ratio_far_rule() -> ValidityRule:
    return ValidityRule(
        id="storage_ratio_far_from_one",
        severity="warning",
        applies=_ratio_is_far,
        message=lambda c: (
            f"The recovery line meets s′ = 0 at t/t′ = {c.value('S_ratio'):.2g} "
            "rather than near 1. Under equation 7 that means the storativity during "
            f"recovery differs from that during pumping (S/S′ = "
            f"{c.value('S_ratio'):.2g}); an aquifer boundary, or a change in pumping "
            "rate before shut-down, produces a similar offset. A change in storage "
            "alone leaves T from the slope unaffected, but check the pumping "
            "duration and the rate history."
        ),
        citation="Theis (1935) eq. 7",
    )


def _storage_ratio_near_rule() -> ValidityRule:
    return ValidityRule(
        id="storage_ratio_near_one",
        severity="info",
        applies=lambda c: not _ratio_is_far(c),
        message=lambda c: (
            f"The recovery line passes close to s′ = 0 at t/t′ = 1 (S/S′ = "
            f"{c.value('S_ratio'):.2f}), as equation 7 expects when storage is the "
            "same during pumping and recovery."
        ),
        citation="Theis (1935) eq. 7",
    )


def _s_not_determined_rule() -> ValidityRule:
    return ValidityRule(
        id="storativity_not_determined",
        severity="info",
        applies=lambda c: True,
        message=lambda c: (
            "Recovery data determine T only. Storativity appears in equation 7 "
            "solely as the ratio S/S′, so S has to come from the pumping phase or "
            "an observation-well analysis."
        ),
        citation="Theis (1935) eq. 7",
    )


def _compare_with_pumping_rule() -> ValidityRule:
    return ValidityRule(
        id="compare_with_pumping_phase",
        severity="info",
        applies=lambda c: True,
        message=lambda c: (
            "Compare this T with the pumping-phase analysis of the same well. "
            "Recovery is free of the pump-rate fluctuations that disturb a drawdown "
            "record, so agreement between the two is the usual check that T is "
            "sound."
        ),
    )
