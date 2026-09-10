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
"""

import logging
from typing import Callable, Optional, Sequence

import numpy as np

from src.calculator.base import BaseCalculator, CalculationInput, CalculationResult
from src.calculator.fitting import (
    DEFAULT_MAXFEV, DEFAULT_TOL, ParameterSpec, covariance_to_ci,
    detect_bound_hits, fit_curve, goodness_of_fit, parameter_correlations,
)
from src.calculator.parameters import FittedParameter
from src.calculator.rules import (
    RuleContext, bound_hit_rule, overfitting_rule, parameter_correlation_rule,
    poor_fit_rule, run_rules,
)
from src.calculator.validation import require_geometry, validate_series

logger = logging.getLogger(__name__)


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
        """Interpretation rules. The three general ones apply to every method."""
        return [
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

        def build(spec):
            # Everything the spec declares, including its display symbol — which
            # used to be dropped, so every template method showed "Sy" and
            # "omega" instead of "S_y" and "ω". Method metadata is merged on top
            # rather than passed alongside, so a method may override any field
            # (symbol included) without a duplicate-keyword error.
            fields = dict(
                name=spec.name,
                value=params[spec.name],
                unit=spec.unit,
                symbol=spec.symbol or spec.name,
                ci=intervals.get(spec.name),
                at_bound=bounds_hit.get(spec.name),
                fixed=not spec.is_free,
                display_scale=spec.display_scale,
                display_unit=spec.display_unit,
            )
            fields.update(meta.get(spec.name, {}))
            return FittedParameter(**fields)

        estimated = [build(spec) for spec in specs]
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
