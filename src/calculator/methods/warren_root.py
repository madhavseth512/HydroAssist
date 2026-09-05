"""Warren-Root (1963) double-porosity fractured aquifer calculator.

Reference: Warren, J.E. & Root, P.J. (1963). The behavior of naturally fractured
reservoirs. Society of Petroleum Engineers Journal, 3(3), 245-255.
(The paper is in the project corpus as `Fractured Aquifers/spe-426-pa.pdf`.)

The first fractured-aquifer method, which completes coverage of all four aquifer
types the registry knows about.

A fractured rock mass behaves as two overlapping media: a fracture network that
transmits nearly all the flow but stores little, and rock matrix blocks that
store most of the water but transmit it slowly. Pumping draws first on the
fractures, then on the matrix as it feeds water into them. The resulting
time-drawdown curve has two parallel Theis-like segments joined by a flatter
transition — and on a derivative plot that transition is an unmistakable dip and
recovery, which is why the diagnostic plot is the right way to spot this aquifer
type in the first place.

Two dimensionless parameters describe it:

    omega   storativity ratio: the fraction of total storage held in the
            fractures. Small omega means the matrix dominates storage and the
            dip is deep.
    lambda  interporosity flow coefficient: how readily the matrix feeds the
            fractures. Small lambda delays the transition.

Both enter through a modified Laplace variable, p -> p*f(p):

    f(p) = (omega*(1-omega)*p + lambda) / ((1-omega)*p + lambda)
    q    = sqrt(p * f(p) * S_total / T)

The exact limit, used as the acceptance test: omega -> 1 puts all storage in the
fractures, there is no matrix to drain, and the solution collapses to Theis.

Assumptions:
  - Fractures are uniformly distributed and connected; the model is a continuum
    approximation, not a discrete fracture network
  - Pseudo-steady-state flow from matrix blocks into fractures
  - Total storativity is the sum of the fracture and matrix contributions

Known limitation worth stating: a double-porosity dip and a leaky-aquifer
derivative can look similar over a short record. Warren-Root recovers to a second
plateau while leakage keeps falling, so the distinction needs late-time data.
"""

import logging

import numpy as np

from src.calculator.fitting import ParameterSpec
from src.calculator.laplace import stehfest_invert_array
from src.calculator.laplace_solutions import q_warren_root, radial_line_source
from src.calculator.parameters import FittedParameter
from src.calculator.rules import (
    ValidityRule, bound_hit_rule, overfitting_rule, parameter_correlation_rule,
    poor_fit_rule,
)
from src.calculator.templates import CurveFitCalculator

logger = logging.getLogger(__name__)

T_MIN, T_MAX = 1e-9, 1.0            # [m²/s]
S_MIN, S_MAX = 1e-9, 1e-1           # total storativity, fractures + matrix

# Storativity ratio. Above ~0.5 the matrix holds less water than the fractures,
# which is not what "double porosity" describes.
OMEGA_MIN, OMEGA_MAX = 1e-5, 1.0

# Interporosity flow coefficient, dimensionless and typically very small.
LAMBDA_MIN, LAMBDA_MAX = 1e-12, 1e-2

# Above this, the fracture and matrix storage are not distinguishable and the
# double-porosity model is not buying anything over Theis.
OMEGA_NEAR_UNITY = 0.5


def warren_root_drawdown(t_arr, T: float, S_total: float, omega: float,
                         lam: float, Q: float, r: float) -> np.ndarray:
    """Drawdown for the Warren-Root double-porosity solution."""
    return stehfest_invert_array(
        lambda p: radial_line_source(
            p, Q, T, r, q_warren_root(p, T, S_total, omega, lam)),
        np.asarray(t_arr, dtype=float),
    )


class WarrenRootCalculator(CurveFitCalculator):
    """Warren-Root (1963) double-porosity analysis for a fractured aquifer.

    Estimates T, total storativity, the storativity ratio omega and the
    interporosity flow coefficient lambda.
    """

    method_id = "warren_root_1963"
    method_label = "Warren-Root (1963)"

    def parameter_specs(self, inputs, time_s, drawdown_m):
        T_init = self._initial_T(time_s, drawdown_m, inputs.Q)
        return [
            ParameterSpec("T", "m^2/s", T_MIN, T_MAX, T_init,
                          symbol="T", log_scale=True,
                          display_scale=86400.0, display_unit="m^2/day",
                          description="Fracture transmissivity"),
            ParameterSpec("S", "-", S_MIN, S_MAX, 1e-4,
                          symbol="S", log_scale=True,
                          description="Total storativity (fractures + matrix)"),
            ParameterSpec("omega", "-", OMEGA_MIN, OMEGA_MAX, 0.05,
                          symbol="ω", log_scale=True,
                          description="Storativity ratio, fracture fraction"),
            ParameterSpec("lambda", "-", LAMBDA_MIN, LAMBDA_MAX, 1e-6,
                          symbol="λ", log_scale=True,
                          description="Interporosity flow coefficient"),
        ]

    def model(self, inputs):
        Q, r = inputs.Q, inputs.r

        def warren_root(t, T, S, omega, lam):
            return warren_root_drawdown(t, T, S, omega, lam, Q, r)

        return warren_root

    @staticmethod
    def _initial_T(time_s, drawdown_m, Q) -> float:
        """From the late-time slope, where the two porosities act as one medium."""
        n = max(3, len(time_s) // 3)
        slope, _ = np.polyfit(np.log10(time_s[-n:]), drawdown_m[-n:], 1)
        if slope > 1e-9:
            estimate = (2.303 * Q) / (4.0 * np.pi * slope)
        else:
            estimate = Q / (4.0 * np.pi * max(float(np.median(drawdown_m)), 1e-6))
        return float(np.clip(estimate, T_MIN * 10, T_MAX / 10))

    def derived_parameters(self, inputs, params):
        """Split total storativity into its fracture and matrix shares — the
        practical question being how much of the resource the rock actually holds."""
        total, omega = params["S"], params["omega"]
        return [
            FittedParameter("S_fracture", total * omega, "-",
                            symbol="S_f", derived=True),
            FittedParameter("S_matrix", total * (1.0 - omega), "-",
                            symbol="S_m", derived=True),
        ]

    def extra_diagnostics(self, inputs, params):
        return {
            "omega": params["omega"],
            "lambda": params["lambda"],
            "storage_contrast": (1.0 - params["omega"]) / max(params["omega"], 1e-12),
        }

    def validity_rules(self):
        return [
            poor_fit_rule(),
            bound_hit_rule(),
            parameter_correlation_rule(),
            overfitting_rule(),
            _omega_near_unity_rule(),
            _transition_visible_rule(),
            _leaky_lookalike_rule(),
        ]


def _omega_near_unity_rule() -> ValidityRule:
    return ValidityRule(
        id="omega_near_unity",
        severity="warning",
        applies=lambda c: c.value("omega", 0.0) > OMEGA_NEAR_UNITY,
        message=lambda c: (
            f"ω = {c.value('omega'):.3f} puts most of the storage in the "
            "fractures, which is the single-porosity case. The matrix is not "
            "contributing measurably, so this reduces to Theis (1935) with two "
            "unnecessary parameters."
        ),
        citation="Warren & Root (1963)",
    )


def _transition_visible_rule() -> ValidityRule:
    """lambda pinned at a bound means the transition fell outside the record."""
    return ValidityRule(
        id="interporosity_transition_not_observed",
        severity="warning",
        applies=lambda c: c.bound_hits.get("lambda") is not None,
        message=lambda c: (
            f"λ reached its {c.bound_hits['lambda']} bound, which means the "
            "fracture-to-matrix transition happened outside the observed time "
            "range. ω and λ are then extrapolated rather than measured; only T "
            "should be relied on."
        ),
    )


def _leaky_lookalike_rule() -> ValidityRule:
    return ValidityRule(
        id="double_porosity_vs_leakage",
        severity="info",
        applies=lambda c: True,
        message=lambda c: (
            "A double-porosity dip and a leaky-aquifer response look alike on a "
            "short record. The distinction is at late time: double porosity "
            "recovers to a second parallel straight line, while leakage keeps "
            "flattening toward a steady state. Compare against Hantush-Jacob "
            "(1955) if the site geology allows either."
        ),
    )
