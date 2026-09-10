"""Template calculators: the fixed part of every method, written once.

A curve-fitting method currently repeats the same eight steps — validate, build
initial estimates, assemble bounds, fit, guard the covariance, convert to
confidence intervals, compute goodness of fit, generate validity notes — and
differs only in its model function, its parameters, and its interpretation rules.
Fifteen more methods means fifteen more copies of the fixed part.

`CurveFitCalculator` runs that sequence and asks a subclass for three things:

    parameter_specs(inputs, t, s)   what is being estimated, with bounds and a
                                    starting point
    model(inputs)                   f(t, *params) -> drawdown
    validity_rules()                how to interpret the result

Everything else — including the two diagnostics no current method has, parameter
correlation and AICc — arrives for free. Hantush-Jacob against this base should
be about 120 lines including its docstring; if it is not, the abstraction is
wrong and should be fixed before fourteen more methods are built on it.

`BaseCalculator` is untouched: these are subclasses, and a method with genuinely
different mechanics can still implement `calculate` directly.

`StraightLineCalculator` is the second template, for methods that are a
regression on a semi-log plot rather than a curve fit: Cooper-Jacob now, and
Theis recovery, Birsoy-Summers step-drawdown, Hvorslev and Bouwer-Rice next.
Point selection, the line's standard errors and the result assembly are the
fixed part; a method supplies the conversion from slope and intercept to its
parameters, and its own rules.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from src.calculator.base import BaseCalculator, CalculationInput, CalculationResult
from src.calculator.fitting import (
    DEFAULT_MAXFEV, DEFAULT_TOL, ParameterSpec, covariance_to_ci,
    detect_bound_hits, fit_curve, goodness_of_fit, parameter_correlations,
)
from src.calculator.parameters import FittedParameter
from src.calculator.rules import (
    RuleContext, ValidityFinding, bound_hit_rule, ci_unavailable_rule,
    overfitting_rule, parameter_correlation_rule, poor_fit_rule, run_rules,
)
from src.calculator.validation import require_geometry, validate_series

logger = logging.getLogger(__name__)


def _build_parameter(spec: ParameterSpec, value: float, ci, at_bound,
                     overrides: dict) -> FittedParameter:
    """A FittedParameter carrying everything its spec declares.

    The display symbol is included — it used to be dropped, so every template
    method showed "Sy" and "omega" instead of "S_y" and "ω". Method metadata is
    merged on top rather than passed alongside, so a method may override any
    field (symbol included) without a duplicate-keyword error.
    """
    fields = dict(
        name=spec.name,
        value=value,
        unit=spec.unit,
        symbol=spec.symbol or spec.name,
        ci=ci,
        at_bound=at_bound,
        fixed=not spec.is_free,
        display_scale=spec.display_scale,
        display_unit=spec.display_unit,
    )
    fields.update(overrides)
    return FittedParameter(**fields)


class CurveFitCalculator(BaseCalculator):
    """Nonlinear least-squares template.

    Subclasses set the class attributes and implement `parameter_specs` and
    `model`. `calculate` is deliberately not meant to be overridden.
    """

    # ── Identity, mirrored from the registry entry ────────────────────────────
    method_id: str = ""
    method_label: str = ""

    # Dotted geometry paths the method cannot run without, e.g. ("well.r_w",).
    requires: tuple = ()

    # Tightened from scipy's 1e-8 default because T and S span four to five
    # orders of magnitude, where the default can converge with a 20% error in T.
    fit_tolerance: float = DEFAULT_TOL
    max_evals: int = DEFAULT_MAXFEV

    # Minimum points before a fit is meaningfully constrained.
    min_points: int = 10

    # ── Subclass contract ─────────────────────────────────────────────────────

    def parameter_specs(self, inputs: CalculationInput,
                        time_s: np.ndarray,
                        drawdown_m: np.ndarray) -> Sequence[ParameterSpec]:
        """What this method estimates. Order defines the model's argument order."""
        raise NotImplementedError

    def model(self, inputs: CalculationInput) -> Callable:
        """f(t, *params) -> drawdown, with params in `parameter_specs` order."""
        raise NotImplementedError

    def validity_rules(self) -> list:
        """Interpretation rules. The general ones apply to every method."""
        return [
            ci_unavailable_rule(),
            poor_fit_rule(),
            bound_hit_rule(),
            parameter_correlation_rule(),
            overfitting_rule(),
        ]

    def extra_validate(self, inputs: CalculationInput) -> Optional[str]:
        """Method-specific input checks beyond the shared ones."""
        return None

    def extra_diagnostics(self, inputs: CalculationInput, params: dict) -> dict:
        """Method-specific values worth publishing, e.g. alpha or t0.

        These end up on `result.diagnostics`, which is how the UI and the report
        stop recomputing quantities the calculator already knew.
        """
        return {}

    def derived_parameters(self, inputs: CalculationInput,
                           params: dict) -> list:
        """Parameters computed from the fitted ones, e.g. T = K*b.

        Returned with derived=True so the report can show them without implying
        they were measured.
        """
        return []

    def parameter_metadata(self) -> dict:
        """Per-parameter presentation and reliability, keyed by name.

        Each entry may carry symbol, display_scale, display_unit, reliable and
        reliability_note — the last two being how a method states a caveat its
        source paper insists on, such as Papadopulos-Cooper on storativity.
        """
        return {}

    # ── The fixed sequence ────────────────────────────────────────────────────

    def calculate(self, inputs: CalculationInput) -> CalculationResult:
        label = self.method_label or self.__class__.__name__

        error = (validate_series(inputs, min_points=self.min_points)
                 or require_geometry(inputs, self.requires, label)
                 or self.extra_validate(inputs))
        if error:
            return CalculationResult(method=label, method_id=self.method_id,
                                     success=False, error_message=error)

        time_s, drawdown_m = inputs.get_series()

        specs = list(self.parameter_specs(inputs, time_s, drawdown_m))
        outcome = fit_curve(
            self.model(inputs), time_s, drawdown_m, specs,
            tol=self.fit_tolerance, maxfev=self.max_evals,
        )
        if not outcome.success:
            return CalculationResult(
                method=label, method_id=self.method_id, success=False,
                error_message=(
                    f"Curve fitting failed: {outcome.message}. "
                    "Check the pumping rate, the well distance and the data quality."
                ),
            )

        params = dict(outcome.values)
        intervals = covariance_to_ci(outcome, specs)
        bounds_hit = detect_bound_hits(outcome, specs)
        correlations = parameter_correlations(outcome)

        fitted = self.model(inputs)(time_s, *[params[s.name] for s in specs])
        gof = goodness_of_fit(drawdown_m, fitted,
                              n_params=len([s for s in specs if s.is_free]))

        meta = self.parameter_metadata()
        estimated = [
            _build_parameter(spec, params[spec.name], intervals.get(spec.name),
                             bounds_hit.get(spec.name), meta.get(spec.name, {}))
            for spec in specs
        ]
        estimated += self.derived_parameters(inputs, params)

        diagnostics = self.extra_diagnostics(inputs, params)
        findings = run_rules(self.validity_rules(), RuleContext(
            params=params,
            specs={s.name: s for s in specs},
            gof=gof,
            time_s=time_s,
            drawdown_obs=drawdown_m,
            drawdown_fitted=fitted,
            bound_hits=bounds_hit,
            correlations=correlations,
            diagnostics=diagnostics,
            inputs=inputs,
            intervals=intervals,
        ))

        logger.info(
            "%s fit: %s R²=%.4f RMSE=%.4fm", label,
            " ".join(f"{k}={v:.3e}" for k, v in params.items()),
            gof.r_squared, gof.rmse,
        )

        return CalculationResult(
            method=label,
            method_id=self.method_id,
            parameters=estimated,
            gof=gof,
            diagnostics=diagnostics,
            findings=findings,
            time_s=time_s,
            drawdown_obs=drawdown_m,
            drawdown_fitted=fitted,
            success=True,
        )


# ── Straight-line template ────────────────────────────────────────────────────

@dataclass
class LineUncertainty:
    """Standard errors of a fitted line and the t critical value that scales them."""

    slope: float
    intercept: float
    se_slope: float
    se_intercept: float
    t_crit: float

    @property
    def slope_interval(self) -> tuple:
        half = self.t_crit * self.se_slope
        return self.slope - half, self.slope + half

    @property
    def intercept_interval(self) -> tuple:
        half = self.t_crit * self.se_intercept
        return self.intercept - half, self.intercept + half


@dataclass
class LineFit:
    """y = slope·x + intercept through the selected points.

    `t` keeps the original times alongside the transformed abscissa, because
    validity criteria are stated in time (u = r²S/4Tt), not in log time.
    """

    slope: float
    intercept: float
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray

    @property
    def n(self) -> int:
        return int(self.x.size)

    def predict(self, x) -> np.ndarray:
        return self.slope * np.asarray(x, dtype=float) + self.intercept

    def uncertainty(self, confidence: float = 0.95):
        """(LineUncertainty, "") or (None, reason).

        The reason is "too_few_points" (fewer than three, so no residual degrees
        of freedom) or "no_spread" (every point at the same abscissa, so the
        slope is undetermined). The t distribution is used rather than the
        normal because a straight-line method often fits only a handful of
        late-time points.
        """
        n = self.n
        if n < 3:
            return None, "too_few_points"

        residuals = self.y - self.predict(self.x)
        s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))
        sxx = np.sum((self.x - np.mean(self.x)) ** 2)
        if sxx < 1e-14:
            return None, "no_spread"

        from scipy.stats import t as t_dist

        quantile = 0.975 if confidence == 0.95 else 0.5 + confidence / 2.0
        return LineUncertainty(
            slope=self.slope,
            intercept=self.intercept,
            se_slope=s_err / np.sqrt(sxx),
            se_intercept=s_err * np.sqrt(np.sum(self.x ** 2) / (n * sxx)),
            t_crit=t_dist.ppf(quantile, df=n - 2),
        ), ""


@dataclass
class Selection:
    """Which points a straight-line method fits, and what it has to say about
    the choice. A non-empty `error` stops the analysis with that message."""

    mask: Optional[np.ndarray] = None     # None means every point
    notes: list = field(default_factory=list)
    error: str = ""


class StraightLineCalculator(BaseCalculator):
    """Semi-log straight-line template.

    The fixed sequence is validate → select the points that belong on the line
    → regress → convert to parameters → clip to bounds → intervals → goodness
    of fit → rules. A subclass supplies:

        parameter_specs(inputs)                units, bounds and display
        line_to_parameters(inputs, line)       {name: value}, or None when the
                                               line is physically meaningless
        select_points(inputs, t, s)            default: every point
        confidence_intervals(inputs, line, params)
                                               ({name: (lo, hi) | None}, notes)
        validity_rules()                       interpretation
    """

    method_id: str = ""
    method_label: str = ""
    requires: tuple = ()
    min_points: int = 10

    # Shown when line_to_parameters rejects the final line.
    line_failure_message: str = (
        "Regression failed: the fitted line is not physically meaningful."
    )

    # ── Subclass contract ─────────────────────────────────────────────────────

    def parameter_specs(self, inputs: CalculationInput) -> Sequence[ParameterSpec]:
        raise NotImplementedError

    def line_to_parameters(self, inputs: CalculationInput,
                           line: LineFit) -> Optional[dict]:
        raise NotImplementedError

    def select_points(self, inputs: CalculationInput, time_s: np.ndarray,
                      drawdown_m: np.ndarray) -> Selection:
        return Selection()

    def confidence_intervals(self, inputs: CalculationInput, line: LineFit,
                             params: dict) -> tuple:
        return {}, []

    def validity_rules(self) -> list:
        return [bound_hit_rule()]

    def validate_inputs(self, inputs: CalculationInput) -> Optional[str]:
        """Checks on the raw series. The shared ones by default; a recovery
        method replaces them, because its residual drawdown may legitimately dip
        below zero once the water level is back at static."""
        return validate_series(inputs, min_points=self.min_points)

    def prepare_series(self, inputs: CalculationInput, time_s: np.ndarray,
                       drawdown_m: np.ndarray) -> tuple:
        """(time, drawdown, notes) the line is analysed in. As given by default;
        a recovery method converts the clock to time since pumping stopped and
        drops the pumping phase. Raise ValueError with a user-facing message
        when nothing usable remains."""
        return time_s, drawdown_m, []

    def extra_validate(self, inputs: CalculationInput) -> Optional[str]:
        return None

    def extra_diagnostics(self, inputs: CalculationInput, line: LineFit,
                          params: dict) -> dict:
        return {}

    def parameter_metadata(self) -> dict:
        return {}

    # ── Shared helpers ────────────────────────────────────────────────────────

    def abscissa(self, inputs: CalculationInput, time_s: np.ndarray) -> np.ndarray:
        """The x axis of the straight-line plot: log10(t) for drawdown methods,
        log10(t/t') for recovery — which is why it sees the inputs."""
        return np.log10(time_s)

    def fit_line(self, inputs: CalculationInput, time_s: np.ndarray,
                 drawdown_m: np.ndarray) -> LineFit:
        time_s = np.asarray(time_s, dtype=float)
        drawdown_m = np.asarray(drawdown_m, dtype=float)
        x = self.abscissa(inputs, time_s)
        slope, intercept = np.polyfit(x, drawdown_m, 1)
        return LineFit(float(slope), float(intercept), x, drawdown_m, time_s)

    # ── The fixed sequence ────────────────────────────────────────────────────

    def calculate(self, inputs: CalculationInput) -> CalculationResult:
        label = self.method_label or self.__class__.__name__

        def failed(message: str) -> CalculationResult:
            return CalculationResult(method=label, method_id=self.method_id,
                                     success=False, error_message=message)

        error = (self.validate_inputs(inputs)
                 or require_geometry(inputs, self.requires, label)
                 or self.extra_validate(inputs))
        if error:
            return failed(error)

        try:
            time_s, drawdown_m, preparation_notes = self.prepare_series(
                inputs, *inputs.get_series())
        except ValueError as exc:
            return failed(str(exc))

        selection = self.select_points(inputs, time_s, drawdown_m)
        if selection.error:
            return failed(selection.error)
        mask = (np.ones(time_s.size, dtype=bool) if selection.mask is None
                else np.asarray(selection.mask, dtype=bool))
        if mask.sum() < 2:
            return failed("Too few points were selected to fit a straight line.")

        # Always refitted on the final selection, so the parameters, the plotted
        # line and the goodness of fit describe one and the same line.
        line = self.fit_line(inputs, time_s[mask], drawdown_m[mask])
        raw = self.line_to_parameters(inputs, line)
        if raw is None:
            return failed(self.line_failure_message)

        specs = list(self.parameter_specs(inputs))
        params, bound_hits = {}, {}
        for spec in specs:
            value = spec.clip(raw[spec.name])
            params[spec.name] = value
            bound_hits[spec.name] = (
                None if value == raw[spec.name]
                else "lower" if value <= spec.lower else "upper"
            )

        intervals, interval_notes = self.confidence_intervals(inputs, line, params)

        on_line = line.predict(line.x)
        gof = goodness_of_fit(line.y, on_line,
                              n_params=len([s for s in specs if s.is_free]))
        diagnostics = self.extra_diagnostics(inputs, line, params)

        # Selection and interval notes first, then the rules — the order a
        # reader needs them: which data were used, how certain the numbers are,
        # then what they mean.
        findings = [ValidityFinding("series_preparation", "info", note)
                    for note in preparation_notes]
        findings += [ValidityFinding("point_selection", "info", note)
                     for note in selection.notes]
        findings += [ValidityFinding("interval_unavailable", "warning", note)
                     for note in interval_notes]
        findings += run_rules(self.validity_rules(), RuleContext(
            params=params,
            specs={s.name: s for s in specs},
            gof=gof,
            time_s=line.t,
            drawdown_obs=line.y,
            drawdown_fitted=on_line,
            bound_hits=bound_hits,
            diagnostics=diagnostics,
            inputs=inputs,
            intervals=intervals,
        ))

        meta = self.parameter_metadata()
        parameters = [
            _build_parameter(spec, params[spec.name], intervals.get(spec.name),
                             bound_hits[spec.name], meta.get(spec.name, {}))
            for spec in specs
        ]

        logger.info(
            "%s line: %s R²=%.4f RMSE=%.4fm on %d/%d points", label,
            " ".join(f"{k}={v:.3e}" for k, v in params.items()),
            gof.r_squared, gof.rmse, line.n, time_s.size,
        )

        # `gof` is deliberately not attached. RMSE and R² above describe the line
        # on the points it was fitted to, which is what the method reports and
        # what the golden baseline pins. Leaving `gof` unset makes
        # CalculationResult compute the model-comparison statistics over EVERY
        # observed point instead — the only basis on which a straight-line
        # method's AICc can be compared with a full-curve fit's.
        return CalculationResult(
            method=label,
            method_id=self.method_id,
            parameters=parameters,
            diagnostics=diagnostics,
            findings=findings,
            rmse=gof.rmse,
            r_squared=gof.r_squared,
            time_s=time_s,
            drawdown_obs=drawdown_m,
            drawdown_fitted=line.predict(self.abscissa(inputs, time_s)),
            success=True,
        )
