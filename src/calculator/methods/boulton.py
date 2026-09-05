"""Boulton (1963) unconfined aquifer calculator with delayed yield.

Reference: Boulton, N.S. (1963). Analysis of data from non-equilibrium pumping
tests allowing for delayed yield from storage. Proceedings of the Institution of
Civil Engineers, 26(3), 469-482.

The first unconfined method, and the first four-parameter fit — which makes it
the first place the parameter-correlation diagnostic earns its keep.

An unconfined aquifer drains in two stages, and its time-drawdown curve has three
segments as a result:

    early    water comes from elastic compression, as if confined; the curve
             follows Theis with the small confined storativity S
    middle   the water table starts to drain under gravity, which supplies water
             and flattens the curve — a plateau that can last a log cycle or more
    late     drainage has caught up, and the curve follows Theis again with the
             much larger specific yield Sy

Boulton represents that middle stage with an empirical drainage constant alpha,
which adds a pole to the characteristic root:

    q = sqrt( (p/T) * (S + Sy*alpha/(p + alpha)) )

The storage term interpolates between S at large p (early time) and S + Sy at
small p (late time), with alpha setting where the transition happens. Two exact
limits follow, and both are acceptance tests:

    alpha -> infinity   instantaneous drainage, so Theis with S + Sy
    alpha -> 0          no drainage on the test timescale, so Theis with S

Known limitation, stated because it decides when NOT to use this method: alpha is
a fitting constant, not a measurable aquifer property. Neuman (1972, 1974)
derived the same three-segment behaviour from the physics of vertical flow in the
unsaturated zone, with no empirical constant, and is the better model where the
data support it. Boulton remains useful because it is far cheaper to evaluate and
converges more reliably.

Assumptions, beyond the Theis set:
  - Unconfined aquifer with a free water table
  - Drawdown small relative to saturated thickness, so transmissivity is constant
  - Fully penetrating well; vertical flow components are absorbed into alpha
    rather than modelled
"""

import logging

import numpy as np

from src.calculator.fitting import ParameterSpec
from src.calculator.laplace import stehfest_invert_array
from src.calculator.laplace_solutions import q_boulton, radial_line_source
from src.calculator.parameters import FittedParameter
from src.calculator.rules import (
    ValidityRule, bound_hit_rule, overfitting_rule, parameter_correlation_rule,
    poor_fit_rule,
)
from src.calculator.templates import CurveFitCalculator

logger = logging.getLogger(__name__)

T_MIN, T_MAX = 1e-9, 1.0        # [m²/s]

# Early-time (elastic) storativity, the same confined range as Theis.
S_MIN, S_MAX = 1e-9, 1e-2

# Specific yield is gravity drainage from pore space. Theis (1935) puts it at
# 0.01-0.35; the upper bound here allows a little headroom for clean gravels.
SY_MIN, SY_MAX = 5e-3, 0.4

# Boulton's drainage constant [1/s]. The reciprocal is the delay index, which
# ranges from minutes in coarse sand to weeks in silty material.
ALPHA_MIN, ALPHA_MAX = 1e-9, 1e-1

# Below this ratio the two storage terms are not separable, and the fit is really
# just Theis with extra parameters along for the ride.
MIN_SY_OVER_S = 10.0


def boulton_drawdown(t_arr, T: float, S: float, Sy: float, alpha: float,
                     Q: float, r: float) -> np.ndarray:
    """Drawdown for Boulton's delayed-yield solution, by Laplace inversion."""
    return stehfest_invert_array(
        lambda p: radial_line_source(p, Q, T, r, q_boulton(p, T, S, Sy, alpha)),
        np.asarray(t_arr, dtype=float),
    )


class BoultonCalculator(CurveFitCalculator):
    """Boulton (1963) delayed-yield analysis for an unconfined aquifer.

    Estimates T, the early elastic storativity S, the specific yield Sy, and the
    drainage constant alpha.
    """

    method_id = "boulton_1963"
    method_label = "Boulton (1963)"

    def parameter_specs(self, inputs, time_s, drawdown_m):
        # All four on a log scale. This is the case log-scale fitting was built
        # for: T spans nine orders of magnitude, alpha eight, and Sy barely two,
        # so a linear parametrisation leaves the Jacobian hopelessly scaled and
        # the optimiser stalls before it finds the delayed-yield plateau.
        T_init = self._initial_T(time_s, drawdown_m, inputs.Q)

        # Start alpha near the middle of the record: the delayed-yield transition
        # is what the parameter describes, so the middle of the observed time
        # range is the least-committal guess available.
        t_mid = float(np.sqrt(time_s.min() * time_s.max()))
        alpha_init = float(np.clip(1.0 / t_mid, ALPHA_MIN, ALPHA_MAX))

        return [
            ParameterSpec("T", "m^2/s", T_MIN, T_MAX, T_init,
                          symbol="T", log_scale=True,
                          display_scale=86400.0, display_unit="m^2/day",
                          description="Transmissivity"),
            ParameterSpec("S", "-", S_MIN, S_MAX, 1e-4,
                          symbol="S", log_scale=True,
                          description="Early-time elastic storativity"),
            ParameterSpec("Sy", "-", SY_MIN, SY_MAX, 0.15,
                          symbol="S_y", log_scale=True,
                          plausible_range=(0.01, 0.35),
                          description="Specific yield"),
            ParameterSpec("alpha", "1/s", ALPHA_MIN, ALPHA_MAX, alpha_init,
                          symbol="α", log_scale=True,
                          description="Boulton drainage constant"),
        ]

    def model(self, inputs):
        Q, r = inputs.Q, inputs.r

        def boulton(t, T, S, Sy, alpha):
            return boulton_drawdown(t, T, S, Sy, alpha, Q, r)

        return boulton

    @staticmethod
    def _initial_T(time_s, drawdown_m, Q) -> float:
        """Cooper-Jacob slope over the latest third, where the curve has resumed
        its Theis behaviour with Sy."""
        n = max(3, len(time_s) // 3)
        slope, _ = np.polyfit(np.log10(time_s[-n:]), drawdown_m[-n:], 1)
        if slope > 1e-9:
            estimate = (2.303 * Q) / (4.0 * np.pi * slope)
        else:
            estimate = Q / (4.0 * np.pi * max(float(np.median(drawdown_m)), 1e-6))
        return float(np.clip(estimate, T_MIN * 10, T_MAX / 10))

    def derived_parameters(self, inputs, params):
        """The delay index 1/alpha is what Boulton's own tables are indexed by,
        and it is a time a hydrogeologist can compare against the test duration."""
        return [
            FittedParameter(
                name="delay_index",
                value=1.0 / params["alpha"],
                unit="s",
                symbol="1/α",
                derived=True,
                display_scale=1.0 / 60.0,
                display_unit="min",
            )
        ]

    def extra_diagnostics(self, inputs, params):
        return {
            "sy_over_s": params["Sy"] / params["S"],
            "delay_index_s": 1.0 / params["alpha"],
            "test_duration_s": float(inputs.get_series()[0].max()),
        }

    def validity_rules(self):
        return [
            poor_fit_rule(),
            bound_hit_rule(),
            parameter_correlation_rule(),
            overfitting_rule(),
            _storage_ratio_rule(),
            _delay_index_vs_duration_rule(),
            _empirical_alpha_rule(),
        ]


def _storage_ratio_rule() -> ValidityRule:
    """Sy and S too close together means delayed yield was never really observed."""
    return ValidityRule(
        id="storage_ratio_too_small",
        severity="warning",
        applies=lambda c: c.diagnostics.get("sy_over_s", 1e9) < MIN_SY_OVER_S,
        message=lambda c: (
            f"Sy/S = {c.diagnostics['sy_over_s']:.1f} is below {MIN_SY_OVER_S:.0f}. "
            "In a real unconfined aquifer specific yield exceeds elastic "
            "storativity by two to four orders of magnitude, so the two storage "
            "terms are not separable here — the extra parameters are not earning "
            "their place, and Theis (1935) would describe this data as well."
        ),
    )


def _delay_index_vs_duration_rule() -> ValidityRule:
    """A delay index beyond the test length means the plateau was never reached."""
    return ValidityRule(
        id="delay_index_exceeds_test",
        severity="warning",
        applies=lambda c: (c.diagnostics.get("delay_index_s", 0)
                           > c.diagnostics.get("test_duration_s", 1e30)),
        message=lambda c: (
            f"The fitted delay index 1/α = {c.diagnostics['delay_index_s'] / 60:.0f} "
            f"min is longer than the test itself "
            f"({c.diagnostics['test_duration_s'] / 60:.0f} min). The delayed-yield "
            "segment was never observed, so Sy and α rest on extrapolation rather "
            "than on data. A longer test is the only real fix."
        ),
    )


def _empirical_alpha_rule() -> ValidityRule:
    return ValidityRule(
        id="alpha_is_empirical",
        severity="info",
        applies=lambda c: True,
        message=lambda c: (
            "Boulton's α is a fitting constant, not a measurable aquifer "
            "property — two sites with identical physics can need different "
            "values. Neuman (1972, 1974) derives the same three-segment "
            "behaviour from vertical flow in the unsaturated zone without an "
            "empirical constant, and is preferable where the data support it."
        ),
        citation="Neuman (1972), critique of the delayed-yield formulation",
    )
