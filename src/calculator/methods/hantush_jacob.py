"""Hantush-Jacob (1955) leaky confined aquifer calculator.

Reference: Hantush, M.S. & Jacob, C.E. (1955). Non-steady radial flow in an
infinite leaky aquifer. Transactions of the American Geophysical Union, 36(1),
95-100.

The first method for an aquifer type other than non-leaky confined, and the first
built entirely on the shared machinery — the Laplace kernel, the fitting
template, the rule library. Its whole numerical content is one characteristic
root.

Key equations:

    s = Q/(4πT) · W(u, r/B)          drawdown, paper eq. 7
    u = r²S / (4Tt)                  same dimensionless time as Theis
    B = √(T·b'/K')                   leakage factor [m]

B is the length over which leakage through the aquitard becomes significant:
large B means a tight confining layer and near-Theis behaviour, small B means a
leaky one. The fitted quantity here is B itself; b' and K' cannot be separated
from drawdown data alone, only their ratio through B.

In the Laplace domain the solution is the same radial line source as Theis with
a modified root — leakage adds a constant term that never vanishes with p:

    q = √(pS/T + 1/B²)      versus Theis   q = √(pS/T)

Two exact limits follow, and both are used as acceptance tests:

    B → ∞      no leakage, and the solution reduces to Theis
    t → ∞      a steady state that Theis never reaches:
               s_max = Q/(2πT) · K₀(r/B)          paper eq. 9

That second limit is the diagnostic signature of a leaky aquifer: drawdown
flattens instead of continuing to grow logarithmically, because the water
eventually comes through the aquitard rather than from aquifer storage.

Assumptions (paper p. 95), beyond the Theis set:
  - The aquitard releases water instantaneously; it has no storage of its own.
    When aquitard storage matters, use Hantush (1960) instead.
  - Leakage is vertical and proportional to the head difference across the
    aquitard.
  - The source bed above the aquitard maintains constant head.
"""

import logging
from typing import Optional

import numpy as np
from scipy.special import kv

from src.calculator.base import CalculationInput
from src.calculator.fitting import ParameterSpec
from src.calculator.laplace import stehfest_invert_array
from src.calculator.laplace_solutions import q_hantush_jacob, radial_line_source
from src.calculator.parameters import FittedParameter
from src.calculator.rules import (
    ValidityRule, bound_hit_rule, confined_storativity_rule, overfitting_rule,
    parameter_correlation_rule, poor_fit_rule,
)
from src.calculator.templates import CurveFitCalculator

logger = logging.getLogger(__name__)

# Same physical envelope as the confined methods.
T_MIN, T_MAX = 1e-9, 1.0        # [m²/s]
S_MIN, S_MAX = 1e-9, 1e-2       # [dimensionless]

# Leakage factor. Below ~1 m the aquitard is so leaky the confined assumption is
# meaningless; above ~1e7 m leakage is undetectable and Theis is the right model.
B_MIN, B_MAX = 1.0, 1e7         # [m]

# r/B below this is indistinguishable from no leakage at all (Hantush's own
# type curves converge on the Theis curve here).
NEGLIGIBLE_R_OVER_B = 0.01

# Above this, leakage dominates so early that T and S are poorly constrained.
STRONG_R_OVER_B = 1.0


def hantush_jacob_drawdown(t_arr, T: float, S: float, B: float,
                           Q: float, r: float) -> np.ndarray:
    """Drawdown from the Hantush-Jacob well function, by Laplace inversion.

    Module-level so plots and the report can draw the curve without constructing
    the calculator.
    """
    return stehfest_invert_array(
        lambda p: radial_line_source(p, Q, T, r, q_hantush_jacob(p, T, S, B)),
        np.asarray(t_arr, dtype=float),
    )


def steady_state_drawdown(Q: float, T: float, r: float, B: float) -> float:
    """The late-time plateau, paper eq. 9:  s_max = Q/(2πT)·K₀(r/B).

    Exact, and the clearest way to explain a leaky result to a user: this is the
    drawdown the well will stabilise at rather than exceeding indefinitely.
    """
    return float(Q / (2.0 * np.pi * T) * kv(0, r / B))


class HantushJacobCalculator(CurveFitCalculator):
    """Hantush-Jacob (1955) analysis for a leaky confined aquifer.

    Estimates T, S and the leakage factor B from time-drawdown data in an
    observation well.
    """

    method_id = "hantush_jacob_1955"
    method_label = "Hantush-Jacob (1955)"

    @staticmethod
    def _initial_T(time_s: np.ndarray, drawdown_m: np.ndarray, Q: float) -> float:
        """Cooper-Jacob straight-line estimate, taken from the MIDDLE of the record.

        Late-time data are avoided deliberately: in a leaky aquifer drawdown
        flattens toward its steady state, so a late-time slope is shallow and
        would put the initial T far too high.
        """
        n = len(time_s)
        lo, hi = max(1, n // 3), max(3, 2 * n // 3)
        t_mid, s_mid = time_s[lo:hi], drawdown_m[lo:hi]
        if len(t_mid) < 3:
            t_mid, s_mid = time_s, drawdown_m

        slope, _ = np.polyfit(np.log10(t_mid), s_mid, 1)
        if slope > 1e-9:
            estimate = (2.303 * Q) / (4.0 * np.pi * slope)
        else:
            estimate = Q / (4.0 * np.pi * max(float(np.median(drawdown_m)), 1e-6))
        return float(np.clip(estimate, T_MIN * 10, T_MAX / 10))

    def parameter_specs(self, inputs, time_s, drawdown_m):
        # Fitted on a log scale: T spans nine orders of magnitude and B seven,
        # so a linear parametrisation leaves the Jacobian badly scaled next to
        # the others and the optimiser stalls.
        T_init = self._initial_T(time_s, drawdown_m, inputs.Q)
        return [
            ParameterSpec("T", "m^2/s", T_MIN, T_MAX, T_init,
                          symbol="T", log_scale=True,
                          display_scale=86400.0, display_unit="m^2/day",
                          description="Transmissivity"),
            ParameterSpec("S", "-", S_MIN, S_MAX, 1e-4,
                          symbol="S", log_scale=True,
                          plausible_range=(1e-6, 1e-3),
                          description="Storativity"),
            ParameterSpec("B", "m", B_MIN, B_MAX, max(10.0 * inputs.r, B_MIN),
                          symbol="B", log_scale=True,
                          description="Leakage factor sqrt(T b'/K')"),
        ]

    def model(self, inputs):
        Q, r = inputs.Q, inputs.r

        def hantush(t, T, S, B):
            return hantush_jacob_drawdown(t, T, S, B, Q, r)

        return hantush

    def parameter_metadata(self):
        return {
            "B": {
                "reliability_note": (
                    "B = sqrt(T·b'/K') combines aquitard thickness and vertical "
                    "conductivity; drawdown data cannot separate them."
                ),
            },
        }

    def derived_parameters(self, inputs, params):
        """r/B is what the paper's type curves are actually labelled by, so it is
        the number a hydrogeologist will want to compare against them."""
        return [
            FittedParameter(
                name="r_over_B",
                value=inputs.r / params["B"],
                unit="-",
                symbol="r/B",
                derived=True,
            )
        ]

    def extra_diagnostics(self, inputs, params):
        r_over_B = inputs.r / params["B"]
        return {
            "r_over_B": r_over_B,
            "leakage_factor_m": params["B"],
            "steady_state_drawdown_m": steady_state_drawdown(
                inputs.Q, params["T"], inputs.r, params["B"]),
        }

    def validity_rules(self):
        return [
            poor_fit_rule(),
            bound_hit_rule(),
            parameter_correlation_rule(),
            overfitting_rule(),
            confined_storativity_rule(),
            _negligible_leakage_rule(),
            _strong_leakage_rule(),
            _steady_state_rule(),
            _aquitard_storage_rule(),
        ]


# ── Method-specific interpretation ────────────────────────────────────────────

def _negligible_leakage_rule() -> ValidityRule:
    """Leakage too small to detect means the data cannot distinguish this model
    from Theis, and the simpler model should be preferred."""
    return ValidityRule(
        id="leakage_negligible",
        severity="warning",
        applies=lambda c: c.diagnostics.get("r_over_B", 1.0) < NEGLIGIBLE_R_OVER_B,
        message=lambda c: (
            f"r/B = {c.diagnostics['r_over_B']:.4f} is below "
            f"{NEGLIGIBLE_R_OVER_B}, where the Hantush-Jacob type curve is "
            "indistinguishable from the Theis curve. The data show no detectable "
            "leakage, so B is not really constrained — prefer Theis (1935), which "
            "estimates the same T and S with one fewer parameter."
        ),
        citation="Hantush & Jacob (1955)",
    )


def _strong_leakage_rule() -> ValidityRule:
    return ValidityRule(
        id="leakage_dominant",
        severity="warning",
        applies=lambda c: c.diagnostics.get("r_over_B", 0.0) > STRONG_R_OVER_B,
        message=lambda c: (
            f"r/B = {c.diagnostics['r_over_B']:.2f} indicates strong leakage: the "
            "aquifer reaches steady state early, so late-time data carry little "
            "information about storativity. Treat S as poorly constrained and "
            "rely on early-time data for it."
        ),
    )


def _steady_state_rule() -> ValidityRule:
    """The practical consequence a well owner actually cares about."""
    return ValidityRule(
        id="leaky_steady_state",
        severity="info",
        applies=lambda c: "steady_state_drawdown_m" in c.diagnostics,
        message=lambda c: (
            "Leakage through the aquitard means drawdown stabilises rather than "
            "growing indefinitely. Predicted maximum drawdown at this distance is "
            f"{c.diagnostics['steady_state_drawdown_m']:.3f} m "
            "(paper eq. 9, s_max = Q/2πT · K₀(r/B))."
        ),
        citation="Hantush & Jacob (1955) eq. 9",
    )


def _aquitard_storage_rule() -> ValidityRule:
    return ValidityRule(
        id="aquitard_storage_ignored",
        severity="info",
        applies=lambda c: True,
        message=lambda c: (
            "Hantush-Jacob assumes the aquitard releases water instantaneously "
            "and stores none of its own. If the confining layer is thick or "
            "compressible, early-time drawdown will be overestimated and "
            "Hantush (1960), which includes aquitard storage, is the better model."
        ),
        citation="Hantush & Jacob (1955), assumptions",
    )
