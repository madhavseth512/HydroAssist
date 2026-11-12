"""Tracer tests: advection and dispersion read off a breakthrough curve.

A tracer test asks a different question from a pumping test. Drawdown says how
easily water moves through an aquifer; a concentration history says how fast it
actually travels and how much it spreads on the way, which is what a contaminant
or a wellhead protection zone depends on. So these methods estimate velocity (or
travel time), longitudinal dispersivity and, given the inputs, effective porosity
— never T or S.

Two arrangements, because they obey different flow fields:

**Natural gradient — Ogata & Banks (1961).** Tracer moves with ambient, uniform
flow to a well at distance x.

    continuous:  C = C0/2 [erfc((x - vt)/2sqrt(Dt)) + exp(vx/D) erfc((x + vt)/2sqrt(Dt))]
    pulse:       C = A / sqrt(4 pi D t) * exp(-(x - vt)^2 / 4Dt),    A = M/(area * n)

The pulse is the plane-source solution of the same equation (Bear, 1972).

**Forced gradient, radially convergent — Sauty (1980).** Tracer enters at distance
r from a well pumping at Q, and velocity rises as 1/r on the way in. The radial
equation has no closed form, so its breakthrough is approximated by the one with
the same first two moments:

    mean arrival    E[t] = t_c (1 + 2 alpha_L / r),    t_c = pi r^2 b n_e / Q
    relative var.   var(t) / t_c^2 = 8 alpha_L / 3r,    so  P_eff = 3r / (4 alpha_L)

Both follow from the radial equation directly (the variance is Hoopes & Harleman's
1967 result). Using the uniform-flow Peclet number P = r/alpha_L instead would
report a dispersivity wrong by a third, and ignoring the shift in the mean would
overstate porosity by 2 alpha_L/r. tests/unit/test_tracer.py holds this
approximation against a numerical solution of the radial equation.

The readings keep the `drawdown_m` column name through the pipeline. Every layer
between the upload and the plots treats it as "the measured response", and the
column is relabelled wherever a person reads it. Concentration is never converted:
it stays in the unit it was reported in.
"""

import logging
from typing import Callable, Optional

import numpy as np
from scipy.special import erfc, erfcx

from src.calculator.fitting import ParameterSpec
from src.calculator.geometry import CONTINUOUS, PULSE, TracerGeometry
from src.calculator.parameters import FittedParameter
from src.calculator.rules import (
    ValidityRule, bound_hit_rule, ci_unavailable_rule, overfitting_rule,
    parameter_correlation_rule, poor_fit_rule,
)
from src.calculator.templates import CurveFitCalculator

logger = logging.getLogger(__name__)

SECONDS_PER_DAY = 86400.0
SECONDS_PER_HOUR = 3600.0

# The search range around what the curve itself shows. Wide enough that hitting a
# limit means the data do not constrain the parameter, not that the range was tight.
TRAVEL_TIME_MARGIN = 10.0
PECLET_MIN, PECLET_MAX = 0.2, 2000.0

# Below this Peclet number the radial approximation starts to lose accuracy.
RADIAL_PECLET_FLOOR = 10.0

# A tail still above this fraction of the peak means the record stopped too early
# to bound the mass or to pin the dispersivity.
TAIL_FRACTION = 0.10
# Fraction of the plateau a continuous test should reach before C0 is measured
# rather than extrapolated.
PLATEAU_FRACTION = 0.90

# Field-scale rule of thumb (Gelhar et al., 1992): alpha_L is around a tenth of the
# travel distance, with very wide scatter.
GELHAR_RATIO_RANGE = (0.001, 1.0)

EFFECTIVE_POROSITY_RANGE = (0.001, 0.6)

# Concentration units that can be turned into a mass, as g/m^3 per unit. Relative
# concentration carries no mass, so recovery cannot be computed from it.
TO_GRAMS_PER_M3 = {"mg/L": 1.0, "ppm": 1.0, "µg/L": 1e-3, "ppb": 1e-3}


# ── The solutions ─────────────────────────────────────────────────────────────

def step_response(t_R: np.ndarray, peclet: float) -> np.ndarray:
    """Ogata-Banks in dimensionless form: C/C0 at t_R = t/t_mean.

    exp(P) erfc(z) overflows long before it is small, so it is evaluated as
    exp(P - z^2) erfcx(z), whose exponent is never large.
    """
    t_R = np.asarray(t_R, dtype=float)
    width = 2.0 * np.sqrt(t_R / peclet)
    upper = (1.0 + t_R) / width
    exponent = -peclet * (1.0 - t_R) ** 2 / (4.0 * t_R)
    return 0.5 * (erfc((1.0 - t_R) / width) + np.exp(exponent) * erfcx(upper))


def _gaussian_exponent(t_R: np.ndarray, peclet: float) -> np.ndarray:
    # Clipped so an optimiser's trial far from the data underflows cleanly.
    return np.exp(np.maximum(-peclet * (1.0 - t_R) ** 2 / (4.0 * t_R), -700.0))


def natural_pulse(t, v: float, alpha_L: float, amplitude: float, x: float) -> np.ndarray:
    """Plane-source pulse in uniform flow. `amplitude` is M/(area * n)."""
    t = np.asarray(t, dtype=float)
    t_R = v * t / x
    return (amplitude / np.sqrt(4.0 * np.pi * alpha_L * x * t_R)
            * _gaussian_exponent(t_R, x / alpha_L))


def natural_continuous(t, v: float, alpha_L: float, C0: float, x: float) -> np.ndarray:
    return C0 * step_response(v * np.asarray(t, dtype=float) / x, x / alpha_L)


def radial_moments(t_c: float, alpha_L: float, r: float) -> tuple:
    """(mean arrival time, effective Peclet number) of the radial breakthrough."""
    return t_c * (1.0 + 2.0 * alpha_L / r), 3.0 * r / (4.0 * alpha_L)


def radial_pulse(t, t_c: float, alpha_L: float, M_over_Q: float, r: float) -> np.ndarray:
    """Flux concentration at the well. Its time integral is M_over_Q."""
    mean, peclet = radial_moments(t_c, alpha_L, r)
    t_R = np.asarray(t, dtype=float) / mean
    return (M_over_Q / mean * np.sqrt(peclet / (4.0 * np.pi * t_R ** 3))
            * _gaussian_exponent(t_R, peclet))


def radial_continuous(t, t_c: float, alpha_L: float, C0: float, r: float) -> np.ndarray:
    mean, peclet = radial_moments(t_c, alpha_L, r)
    return C0 * step_response(np.asarray(t, dtype=float) / mean, peclet)


_SOLUTIONS = {
    ("ogata_banks_1961", PULSE): natural_pulse,
    ("ogata_banks_1961", CONTINUOUS): natural_continuous,
    ("sauty_1980", PULSE): radial_pulse,
    ("sauty_1980", CONTINUOUS): radial_continuous,
}


def curve_from_result(result, t, first: float, second: float) -> np.ndarray:
    """The modelled concentration at `t`, with the two hand-matching parameters
    moved and the amplitude held at its fit. The result carries what its curve
    needs, so plots and the report never have to re-read the setup form."""
    info = result.curves["tracer"]
    solution = _SOLUTIONS[(result.method_id, info["injection"])]
    amplitude = result.value(info["amplitude"])
    return info["background"] + solution(t, first, second, amplitude, info["distance"])


# ── What the curve itself shows ───────────────────────────────────────────────

def _crossing(t: np.ndarray, y: np.ndarray, level: float, rising: bool = True) -> Optional[float]:
    """The time `y` first crosses `level`, interpolated between readings."""
    crossed = y >= level if rising else y <= level
    indices = np.flatnonzero(crossed)
    if indices.size == 0:
        return None
    i = int(indices[0])
    if i == 0:
        return float(t[0])
    span = y[i] - y[i - 1]
    fraction = (level - y[i - 1]) / span if span else 0.0
    return float(t[i - 1] + fraction * (t[i] - t[i - 1]))


def _pulse_shape(t: np.ndarray, y: np.ndarray) -> tuple:
    """(peak time, peak, Peclet estimate) from the peak's half-height width."""
    peak_index = int(np.argmax(y))
    peak_time, peak = float(t[peak_index]), float(y[peak_index])
    half = peak / 2.0
    start = _crossing(t[:peak_index + 1], y[:peak_index + 1], half)
    end = _crossing(t[peak_index:], y[peak_index:], half, rising=False)
    if start is None or end is None or end <= start:
        return peak_time, peak, 20.0
    sigma = (end - start) / 2.355
    return peak_time, peak, 2.0 * (peak_time / sigma) ** 2


def _step_shape(t: np.ndarray, y: np.ndarray, plateau: float) -> tuple:
    """(midpoint time, Peclet estimate) from the 16% to 84% rise."""
    middle = _crossing(t, y, 0.5 * plateau) or float(np.median(t))
    low = _crossing(t, y, 0.16 * plateau)
    high = _crossing(t, y, 0.84 * plateau)
    if low is None or high is None or high <= low:
        return middle, 20.0
    sigma = (high - low) / 2.0
    return middle, 2.0 * (middle / sigma) ** 2


def _clip_peclet(peclet: float) -> float:
    return float(np.clip(peclet, PECLET_MIN * 5, PECLET_MAX / 5))


# ── The shared calculator ─────────────────────────────────────────────────────

def _tracer_inputs(inputs) -> TracerGeometry:
    return inputs.tracer if inputs.tracer is not None else TracerGeometry()


def _response(inputs, concentration: np.ndarray) -> np.ndarray:
    """Concentration above background, which is what the solutions describe."""
    return np.maximum(concentration - _tracer_inputs(inputs).background, 0.0)


class _TracerCalculator(CurveFitCalculator):
    """What the two arrangements share: checks, starting points, interpretation."""

    needs_rate = False
    fit_failure_message = ("Check the distance, the injection type and that the "
                           "times count from the injection.")

    # ── Checks ────────────────────────────────────────────────────────────────

    def validate_inputs(self, inputs) -> Optional[str]:
        if self.needs_rate and (inputs.Q is None or inputs.Q <= 0):
            return "A forced-gradient tracer test needs the pumping rate Q."
        if inputs.r <= 0:
            return f"The travel distance must be positive (got {inputs.r} m)."

        t, c = inputs.get_series()
        if len(t) < self.min_points:
            return (f"Insufficient data: {len(t)} points "
                    f"(minimum {self.min_points} required).")
        if np.any(t <= 0):
            return "Times must count forward from the injection, starting after zero."
        if np.any(c < 0):
            return ("Negative concentrations detected. Enter readings as measured "
                    "and give the background separately.")
        if np.ptp(c) == 0 or not np.any(_response(inputs, c) > 0):
            return ("These readings show no breakthrough: the concentration never "
                    "rises above the background.")
        return None

    # ── Subclass contract ─────────────────────────────────────────────────────

    def travel_specs(self, inputs, t_mean: float, t_low: float, t_high: float) -> ParameterSpec:
        raise NotImplementedError

    def pulse_amplitude(self, inputs, peak: float, t_mean: float, peclet: float) -> ParameterSpec:
        raise NotImplementedError

    def solution(self, injection: str) -> Callable:
        return _SOLUTIONS[(self.method_id, injection)]

    # ── Parameters ────────────────────────────────────────────────────────────

    def parameter_specs(self, inputs, time_s, drawdown_m):
        tracer = _tracer_inputs(inputs)
        y = _response(inputs, drawdown_m)
        t_low, t_high = (float(time_s.min()) / TRAVEL_TIME_MARGIN,
                         float(time_s.max()) * TRAVEL_TIME_MARGIN)

        if tracer.injection == PULSE:
            t_mean, peak, peclet = _pulse_shape(time_s, y)
            peclet = _clip_peclet(peclet)
            amplitude = self.pulse_amplitude(inputs, peak, t_mean, peclet)
        else:
            plateau = tracer.C0 if tracer.C0 else float(np.max(y))
            t_mean, peclet = _step_shape(time_s, y, plateau)
            peclet = _clip_peclet(peclet)
            amplitude = _plateau_spec(tracer, plateau)

        return [
            self.travel_specs(inputs, t_mean, t_low, t_high),
            ParameterSpec("alpha_L", "m", inputs.r / PECLET_MAX, inputs.r / PECLET_MIN,
                          inputs.r / peclet, symbol="α_L", log_scale=True,
                          description="Longitudinal dispersivity"),
            amplitude,
        ]

    def model(self, inputs):
        tracer = _tracer_inputs(inputs)
        solution = self.solution(tracer.injection)
        distance, background = inputs.r, tracer.background

        def tracer_model(t, travel, alpha_L, amplitude):
            return background + solution(t, travel, alpha_L, amplitude, distance)
        return tracer_model

    def extra_curves(self, inputs, params):
        tracer = _tracer_inputs(inputs)
        return {"tracer": {
            "injection": tracer.injection,
            "background": tracer.background,
            "distance": inputs.r,
            "unit": tracer.conc_unit,
            "amplitude": "C0" if tracer.injection == CONTINUOUS else self.pulse_name,
        }}

    def extra_diagnostics(self, inputs, params):
        t, c = inputs.get_series()
        y = _response(inputs, c)
        peak = float(np.max(y))
        diagnostics = {
            "peclet": inputs.r / params["alpha_L"],
            "peak_time_s": float(t[int(np.argmax(y))]),
            "last_reading_fraction": float(y[-1] / peak) if peak else 0.0,
        }
        if _tracer_inputs(inputs).injection == CONTINUOUS:
            diagnostics["fraction_of_plateau"] = float(y[-1] / params["C0"])
        return diagnostics

    def validity_rules(self):
        return [
            ci_unavailable_rule(),
            bound_hit_rule(),
            parameter_correlation_rule(),
            overfitting_rule(),
            poor_fit_rule(),
            _peak_captured_rule(),
            _tail_rule(),
            _plateau_rule(),
            _dispersivity_scale_rule(),
            _porosity_rule(),
        ]

    # ── Derived quantities ────────────────────────────────────────────────────

    def _derived(self, name, value, unit, symbol, **kw) -> FittedParameter:
        return FittedParameter(name=name, value=float(value), unit=unit,
                               symbol=symbol, derived=True, **kw)


def _plateau_spec(tracer: TracerGeometry, plateau: float) -> ParameterSpec:
    """C0 is pinned when the user knows it, and fitted when they do not."""
    return ParameterSpec(
        "C0", tracer.conc_unit, plateau * 0.5, plateau * 100.0, plateau,
        symbol="C₀", log_scale=True, fixed=tracer.C0 or None,
        description="Plateau concentration",
    )


class OgataBanksCalculator(_TracerCalculator):
    """Natural-gradient tracer test: one-dimensional advection and dispersion in
    uniform flow, with the readings taken a known distance downgradient."""

    method_id = "ogata_banks_1961"
    method_label = "Ogata-Banks (1961)"
    pulse_name = "A_over_n"

    def travel_specs(self, inputs, t_mean, t_low, t_high):
        return ParameterSpec(
            "v", "m/s", inputs.r / t_high, inputs.r / t_low, inputs.r / t_mean,
            symbol="v", log_scale=True, display_scale=SECONDS_PER_DAY,
            display_unit="m/day", description="Average linear velocity")

    def pulse_amplitude(self, inputs, peak, t_mean, peclet):
        unit = f"{_tracer_inputs(inputs).conc_unit}·m"
        initial = peak * np.sqrt(4.0 * np.pi / peclet) * inputs.r
        return ParameterSpec(self.pulse_name, unit, initial * 1e-3, initial * 1e3,
                             initial, symbol="M/(A·n)", log_scale=True,
                             description="Injected mass per unit flow area")

    def derived_parameters(self, inputs, params):
        tracer = _tracer_inputs(inputs)
        v, alpha_L = params["v"], params["alpha_L"]
        derived = [
            self._derived("D_L", alpha_L * v, "m^2/s", "D_L",
                          display_scale=SECONDS_PER_DAY, display_unit="m^2/day"),
            self._derived("Pe", inputs.r / alpha_L, "-", "Pe = x/α_L"),
        ]
        if tracer.K is not None and tracer.gradient is not None:
            # Darcy: v = K i / n_e.
            derived.append(self._derived("n_e", tracer.K * tracer.gradient / v,
                                         "-", "n_e"))
        return derived


class SautyCalculator(_TracerCalculator):
    """Forced-gradient tracer test: tracer drawn radially to a pumped well."""

    method_id = "sauty_1980"
    method_label = "Sauty (1980)"
    needs_rate = True
    pulse_name = "M_over_Q"

    def travel_specs(self, inputs, t_mean, t_low, t_high):
        return ParameterSpec(
            "t_c", "s", t_low, t_high, t_mean, symbol="t_c", log_scale=True,
            display_scale=1.0 / SECONDS_PER_HOUR, display_unit="hours",
            description="Advective travel time")

    def pulse_amplitude(self, inputs, peak, t_mean, peclet):
        unit = f"{_tracer_inputs(inputs).conc_unit}·s"
        initial = peak * t_mean * np.sqrt(4.0 * np.pi / (0.75 * peclet))
        return ParameterSpec(self.pulse_name, unit, initial * 1e-3, initial * 1e3,
                             initial, symbol="M/Q", log_scale=True,
                             description="Recovered mass over pumping rate")

    def derived_parameters(self, inputs, params):
        tracer = _tracer_inputs(inputs)
        t_c, alpha_L = params["t_c"], params["alpha_L"]
        derived = [self._derived("Pe", inputs.r / alpha_L, "-", "Pe = r/α_L")]

        if inputs.aquifer.b is not None:
            # t_c = pi r^2 b n_e / Q, the volume of the cylinder over the rate.
            derived.append(self._derived(
                "n_e", inputs.Q * t_c / (np.pi * inputs.r ** 2 * inputs.aquifer.b),
                "-", "n_e"))

        to_grams = TO_GRAMS_PER_M3.get(tracer.conc_unit)
        if tracer.injection == PULSE and to_grams is not None:
            recovered_kg = params[self.pulse_name] * inputs.Q * to_grams / 1000.0
            derived.append(self._derived("M_recovered", recovered_kg, "kg", "M_rec"))
            if tracer.M_injected is not None:
                derived.append(self._derived(
                    "recovery", recovered_kg / tracer.M_injected, "-",
                    "Mass recovery"))
        return derived

    def validity_rules(self):
        return super().validity_rules() + [_radial_peclet_rule(), _recovery_rule()]


# ── Interpretation ────────────────────────────────────────────────────────────

def _is_pulse(c) -> bool:
    return _tracer_inputs(c.inputs).injection == PULSE


def _peak_captured_rule() -> ValidityRule:
    """A peak at either end of the record is not a peak: the fit is extrapolating
    the one feature that fixes the travel time."""

    def applies(c):
        peak_time = c.diagnostics["peak_time_s"]
        return _is_pulse(c) and peak_time in (c.time_s[0], c.time_s[-1])

    return ValidityRule(
        id="peak_not_captured",
        severity="critical",
        applies=applies,
        message=lambda c: (
            "The highest reading is at the very "
            f"{'start' if c.diagnostics['peak_time_s'] == c.time_s[0] else 'end'} "
            "of the record, so the peak itself was not observed. Travel time and "
            "dispersivity are extrapolated. Check the time origin, or that "
            "sampling lasted long enough."
        ),
    )


def _tail_rule() -> ValidityRule:
    return ValidityRule(
        id="tail_truncated",
        severity="warning",
        applies=lambda c: _is_pulse(c)
        and c.diagnostics["last_reading_fraction"] > TAIL_FRACTION,
        message=lambda c: (
            "Sampling stopped with the concentration still at "
            f"{c.diagnostics['last_reading_fraction']:.0%} of the peak. The tail "
            "was not recorded, so dispersivity and any recovered mass rest on the "
            "model's extrapolation."
        ),
    )


def _plateau_rule() -> ValidityRule:
    return ValidityRule(
        id="plateau_not_reached",
        severity="warning",
        applies=lambda c: (not _is_pulse(c)
                           and c.diagnostics["fraction_of_plateau"] < PLATEAU_FRACTION),
        message=lambda c: (
            "The record ends at "
            f"{c.diagnostics['fraction_of_plateau']:.0%} of the plateau, so the "
            "plateau is extrapolated and trades off against velocity. Enter the "
            "injected concentration C₀ if it is known."
        ),
    )


def _dispersivity_scale_rule() -> ValidityRule:
    def ratio(c):
        return c.value("alpha_L") / c.inputs.r

    return ValidityRule(
        id="dispersivity_scale",
        severity="info",
        applies=lambda c: not (GELHAR_RATIO_RANGE[0] <= ratio(c) <= GELHAR_RATIO_RANGE[1]),
        message=lambda c: (
            f"α_L is {ratio(c):.3g} of the travel distance. Field values are often "
            "near one tenth, with very wide scatter (Gelhar et al., 1992), so check "
            "the distance and the time origin."
        ),
        citation="Gelhar et al. (1992)",
    )


def _porosity_rule() -> ValidityRule:
    def porosity(c):
        if c.inputs.tracer is None:
            return None
        if "t_c" in c.params:
            if c.inputs.aquifer.b is None:
                return None
            return (c.inputs.Q * c.value("t_c")
                    / (np.pi * c.inputs.r ** 2 * c.inputs.aquifer.b))
        tracer = c.inputs.tracer
        if tracer.K is None or tracer.gradient is None:
            return None
        return tracer.K * tracer.gradient / c.value("v")

    def applies(c):
        n_e = porosity(c)
        return (n_e is not None
                and not (EFFECTIVE_POROSITY_RANGE[0] <= n_e <= EFFECTIVE_POROSITY_RANGE[1]))

    return ValidityRule(
        id="porosity_implausible",
        severity="warning",
        applies=applies,
        message=lambda c: (
            f"Effective porosity n_e = {porosity(c):.3g} is outside the usual "
            f"{EFFECTIVE_POROSITY_RANGE[0]}–{EFFECTIVE_POROSITY_RANGE[1]}. Check the "
            "units of the inputs it is computed from. A low value can be genuine in "
            "fractured rock, where flow is confined to a small part of the volume."
        ),
    )


def _radial_peclet_rule() -> ValidityRule:
    return ValidityRule(
        id="radial_peclet_low",
        severity="warning",
        applies=lambda c: c.diagnostics["peclet"] < RADIAL_PECLET_FLOOR,
        message=lambda c: (
            f"Pe = r/α_L = {c.diagnostics['peclet']:.1f}, below "
            f"{RADIAL_PECLET_FLOOR:.0f}. The radial breakthrough approximation "
            "loses accuracy this far from advection-dominated transport, so treat "
            "α_L and n_e as approximate."
        ),
    )


def _recovery_rule() -> ValidityRule:
    def fraction(c):
        tracer = c.inputs.tracer
        to_grams = TO_GRAMS_PER_M3.get(tracer.conc_unit) if tracer else None
        if (tracer is None or tracer.M_injected is None or to_grams is None
                or tracer.injection != PULSE):
            return None
        return (c.value(SautyCalculator.pulse_name) * c.inputs.Q * to_grams
                / 1000.0 / tracer.M_injected)

    def applies(c):
        recovered = fraction(c)
        return recovered is not None and recovered > 1.1

    return ValidityRule(
        id="recovery_above_one",
        severity="warning",
        applies=applies,
        message=lambda c: (
            f"The fit implies {fraction(c):.0%} of the injected mass was recovered, "
            "which is more than was injected. Check Q, the injected mass, the "
            "concentration unit and the background."
        ),
    )
