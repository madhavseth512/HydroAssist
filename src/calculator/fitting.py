"""Shared nonlinear-least-squares machinery for every curve-fitting method.

Theis, Cooper-Jacob and Papadopulos-Cooper each carry their own near-identical
copy of: assemble bounds, call curve_fit, guard the covariance matrix, turn
standard errors into confidence intervals, detect bound hits, compute RMSE and
R-squared. Every new method would add another copy. This module holds one of each.

Two capabilities exist here that the current calculators do not have, both needed
before multi-parameter methods can be trusted:

**Log-scale fitting.** Trust Region Reflective sees T spanning 1e-9 to 1e0
alongside parameters of order 1 (Neuman's beta, Warren-Root's omega). The Jacobian
is then hopelessly scaled and the optimiser stalls or converges to a false
optimum. Fitting log10(T) fixes it. Existing methods keep log_scale=False so their
numbers do not move.

**AIC and AICc.** Once a user can choose between 2-parameter Theis and
6-parameter Moench, R-squared will always favour Moench, because adding parameters
cannot make the residuals worse. Reporting only R-squared would make the app
actively recommend overfitting. Pumping tests typically carry 30-60 points, so
with 4-6 parameters n/k falls well inside the range where plain AIC is biased and
the small-sample-corrected AICc is the right statistic.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.optimize import curve_fit

# Tolerance used by the existing calculators. Tighter than scipy's 1e-8 default
# because T and S span four to five orders of magnitude, where the default can
# declare convergence with a 20% error in T.
DEFAULT_TOL = 1e-12
DEFAULT_MAXFEV = 10000

# 95% two-sided normal critical value. The existing Theis and Papadopulos-Cooper
# calculators use this rather than a t critical value; keeping it as the default
# is what lets those results reproduce exactly when they move onto this module.
Z_95 = 1.96


@dataclass(frozen=True)
class ParameterSpec:
    """One fittable parameter: what it is, where it may live, where to start.

    `name` is the canonical key ("T", "S", "Sy", "B", "beta", "omega"). Order in
    the spec list is the order the model function receives its arguments.
    """

    name: str
    unit: str
    lower: float
    upper: float
    initial: float

    symbol: str = ""
    description: str = ""

    # Fit log10(value) instead of value. Essential for parameters spanning many
    # orders of magnitude when fitted alongside parameters of order 1.
    log_scale: bool = False

    # Pinned by the user rather than estimated. Excluded from the free vector,
    # still passed to the model.
    fixed: Optional[float] = None

    # Display conversion, e.g. T in m^2/s shown as m^2/day.
    display_scale: float = 1.0
    display_unit: str = ""

    # Physically expected range; a value outside it earns a validity note rather
    # than an error, because unusual aquifers exist.
    plausible_range: Optional[tuple] = None

    def __post_init__(self):
        if self.lower >= self.upper:
            raise ValueError(
                f"{self.name}: lower bound {self.lower} must be below upper "
                f"bound {self.upper}."
            )
        if self.log_scale and self.lower <= 0:
            raise ValueError(
                f"{self.name}: log-scale fitting needs a positive lower bound, "
                f"got {self.lower}."
            )

    @property
    def is_free(self) -> bool:
        return self.fixed is None

    def clip(self, value: float) -> float:
        return float(np.clip(value, self.lower, self.upper))


@dataclass
class FitOutcome:
    """Result of one curve_fit call, in natural (not log) units."""

    values: dict                      # name -> fitted value
    covariance: Optional[np.ndarray]  # over FREE parameters, in fitting space
    free_names: list                  # order matching the covariance matrix
    success: bool = True
    message: str = ""

    def ordered(self, specs: Sequence[ParameterSpec]) -> list:
        return [self.values[s.name] for s in specs]


@dataclass
class GoodnessOfFit:
    """Fit quality. AIC/AICc exist so models with different parameter counts can
    be compared without the comparison being rigged in favour of the bigger one."""

    rmse: float
    r_squared: float
    adjusted_r_squared: float
    ss_res: float
    ss_tot: float
    n_points: int
    n_params: int
    aic: float
    aicc: float

    @property
    def aicc_is_reliable(self) -> bool:
        """AICc needs n - k - 1 > 0 to be defined at all."""
        return self.n_points - self.n_params - 1 > 0

    @property
    def prefer_aicc(self) -> bool:
        """Below n/k ~ 40, plain AIC is biased toward the larger model."""
        return self.n_params > 0 and self.n_points / self.n_params < 40


# ── Parameter vector plumbing ─────────────────────────────────────────────────

def _to_fit_space(spec: ParameterSpec, value: float) -> float:
    return float(np.log10(value)) if spec.log_scale else float(value)


def _from_fit_space(spec: ParameterSpec, value: float) -> float:
    return float(10.0 ** value) if spec.log_scale else float(value)


def _assemble(specs: Sequence[ParameterSpec], free_values: Sequence[float]) -> list:
    """Interleave fitted free values with pinned ones, back in natural units."""
    out, i = [], 0
    for spec in specs:
        if spec.is_free:
            out.append(_from_fit_space(spec, free_values[i]))
            i += 1
        else:
            out.append(float(spec.fixed))
    return out


def fit_curve(
    model: Callable,
    x: np.ndarray,
    y: np.ndarray,
    specs: Sequence[ParameterSpec],
    *,
    tol: float = DEFAULT_TOL,
    maxfev: int = DEFAULT_MAXFEV,
    method: str = "trf",
    sigma: Optional[np.ndarray] = None,
) -> FitOutcome:
    """Fit `model(x, *params)` to `y`, with params described by `specs`.

    `model` always receives parameters in natural units and in spec order,
    including any pinned ones — it never needs to know about log scaling or
    which parameters are free.

    With every spec at log_scale=False, no fixed parameters and tol=1e-12, this
    is numerically identical to the curve_fit call the existing calculators make.
    """
    free = [s for s in specs if s.is_free]
    if not free:
        raise ValueError("At least one parameter must be free to fit.")

    def wrapped(t, *free_values):
        return model(t, *_assemble(specs, free_values))

    p0, lower, upper = [], [], []
    for spec in free:
        initial = spec.clip(spec.initial)
        p0.append(_to_fit_space(spec, initial))
        lower.append(_to_fit_space(spec, spec.lower))
        upper.append(_to_fit_space(spec, spec.upper))

    try:
        popt, pcov = curve_fit(
            wrapped, x, y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=maxfev,
            method=method,
            ftol=tol, xtol=tol, gtol=tol,
            sigma=sigma,
        )
    except (RuntimeError, ValueError) as exc:
        return FitOutcome(
            values={s.name: (s.fixed if not s.is_free else s.initial) for s in specs},
            covariance=None,
            free_names=[s.name for s in free],
            success=False,
            message=str(exc),
        )

    natural = _assemble(specs, popt)
    return FitOutcome(
        values={spec.name: value for spec, value in zip(specs, natural)},
        covariance=pcov,
        free_names=[s.name for s in free],
        success=True,
    )


# ── Uncertainty ───────────────────────────────────────────────────────────────

def covariance_to_ci(
    outcome: FitOutcome,
    specs: Sequence[ParameterSpec],
    *,
    confidence: float = 0.95,
    use_t: bool = False,
    n_points: Optional[int] = None,
) -> dict:
    """Confidence intervals per parameter name, in natural units.

    Returns None for a parameter whenever the interval is not trustworthy:
    a pinned parameter, a failed fit, or a covariance diagonal containing inf
    (curve_fit did not converge to a unique solution) or a negative value (an
    ill-conditioned Jacobian). Both of the latter usually mean the data do not
    constrain the model — aquifer heterogeneity, boundary effects, or simply too
    many parameters for the information present.

    For a log-scale parameter the interval is built in log space and
    back-transformed, so it is correctly asymmetric in natural units.
    """
    empty = {spec.name: None for spec in specs}
    if not outcome.success or outcome.covariance is None:
        return empty

    diag = np.diag(outcome.covariance)
    if np.any(np.isinf(diag)) or np.any(diag < 0):
        return empty

    if use_t:
        from scipy.stats import t as t_dist
        dof = (n_points or 0) - len(outcome.free_names)
        if dof <= 0:
            return empty
        crit = float(t_dist.ppf(0.5 + confidence / 2.0, df=dof))
    else:
        crit = Z_95 if confidence == 0.95 else _normal_critical(confidence)

    errors = np.sqrt(diag)
    by_name = dict(zip(outcome.free_names, errors))

    out = {}
    for spec in specs:
        if not spec.is_free or spec.name not in by_name:
            out[spec.name] = None
            continue
        centre = _to_fit_space(spec, outcome.values[spec.name])
        half = crit * by_name[spec.name]
        lo, hi = centre - half, centre + half
        out[spec.name] = (_from_fit_space(spec, lo), _from_fit_space(spec, hi))
    return out


def _normal_critical(confidence: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(0.5 + confidence / 2.0))


def parameter_correlations(outcome: FitOutcome) -> dict:
    """Pairwise correlation between free parameters, from the covariance matrix.

    This is how the dominant failure of 4-6 parameter fits gets caught. Such fits
    usually do not fail to converge — they converge to a non-unique optimum where
    two parameters trade off almost perfectly. curve_fit still returns a finite
    covariance and the fit still looks excellent, but the individual values are
    meaningless. A correlation above about 0.98 is the signature.
    """
    if outcome.covariance is None or len(outcome.free_names) < 2:
        return {}

    cov = np.asarray(outcome.covariance, dtype=float)
    sd = np.sqrt(np.diag(cov))
    if np.any(~np.isfinite(sd)) or np.any(sd <= 0):
        return {}

    out = {}
    for i, a in enumerate(outcome.free_names):
        for j, b in enumerate(outcome.free_names):
            if j <= i:
                continue
            out[(a, b)] = float(cov[i, j] / (sd[i] * sd[j]))
    return out


def detect_bound_hits(
    outcome: FitOutcome,
    specs: Sequence[ParameterSpec],
    *,
    rtol: float = 1e-6,
) -> dict:
    """Which parameters landed on a bound: name -> "lower" | "upper" | None.

    A parameter pinned against a bound means the true value may lie outside the
    range the method allows, so the estimate is a floor or ceiling rather than a
    measurement. Silently returning it as if it were fitted is the bug this
    prevents.
    """
    out = {}
    for spec in specs:
        if not spec.is_free or spec.name not in outcome.values:
            out[spec.name] = None
            continue
        value = outcome.values[spec.name]
        span = abs(spec.upper - spec.lower) or 1.0
        if abs(value - spec.lower) <= rtol * span:
            out[spec.name] = "lower"
        elif abs(value - spec.upper) <= rtol * span:
            out[spec.name] = "upper"
        else:
            out[spec.name] = None
    return out


# ── Goodness of fit ───────────────────────────────────────────────────────────

def goodness_of_fit(y_obs: np.ndarray, y_fit: np.ndarray,
                    n_params: int) -> GoodnessOfFit:
    """RMSE, R-squared and the information criteria.

    The RMSE and R-squared arithmetic matches the block currently duplicated in
    all three calculators, including the ss_tot == 0 guard.
    """
    y_obs = np.asarray(y_obs, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    n = int(y_obs.size)

    residuals = y_obs - y_fit
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    dof = n - n_params - 1
    adjusted = (1.0 - (1.0 - r_squared) * (n - 1) / dof) if dof > 0 else 0.0

    # Gaussian-likelihood AIC up to an additive constant, which cancels when
    # comparing models on the same data.
    if ss_res > 0 and n > 0:
        aic = n * np.log(ss_res / n) + 2 * n_params
    else:
        aic = float("-inf")   # a perfect fit; comparison is meaningless anyway

    denom = n - n_params - 1
    aicc = aic + (2 * n_params * (n_params + 1) / denom) if denom > 0 else float("inf")

    return GoodnessOfFit(
        rmse=rmse,
        r_squared=r_squared,
        adjusted_r_squared=float(adjusted),
        ss_res=ss_res,
        ss_tot=ss_tot,
        n_points=n,
        n_params=int(n_params),
        aic=float(aic),
        aicc=float(aicc),
    )
