"""Validity rules as data.

Every calculator ends with a block that inspects the fit and appends interpretation
notes: is R-squared low, is storativity outside the confined range, did a parameter
hit a bound, should another method be tried. These are the notes the PDF prints and
the consultant reads, and they are the reason a user should trust — or distrust — a
number. Written as free functions per method they cannot be shared, tested in
isolation, or filtered by severity.

Expressing them as `ValidityRule` objects makes each one independently testable,
lets a new method reuse the general ones, and gives every note a stable `id` and a
severity so the UI can rank them instead of printing a flat list.

Rules are pure: they read a RuleContext and return text. They never mutate state.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass
class RuleContext:
    """Everything a rule may inspect about a completed fit."""

    params: dict                       # name -> fitted value, natural units
    specs: dict                        # name -> ParameterSpec
    gof: object                        # GoodnessOfFit
    time_s: np.ndarray
    drawdown_obs: np.ndarray
    drawdown_fitted: np.ndarray
    bound_hits: dict = field(default_factory=dict)      # name -> "lower"|"upper"|None
    correlations: dict = field(default_factory=dict)    # (a, b) -> correlation
    diagnostics: dict = field(default_factory=dict)     # method-specific extras
    inputs: object = None
    # name -> (lo, hi) | None. Empty means intervals were not supplied, which
    # is different from intervals that could not be computed.
    intervals: dict = field(default_factory=dict)

    def value(self, name: str, default: float = 0.0) -> float:
        return self.params.get(name, default)

    def spec(self, name: str):
        return self.specs.get(name)


@dataclass(frozen=True)
class ValidityFinding:
    """One note about a completed fit."""

    id: str
    severity: str            # "info" | "warning" | "critical"
    text: str
    citation: str = ""

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class ValidityRule:
    """A condition and the note it produces."""

    id: str
    severity: str
    applies: Callable            # RuleContext -> bool
    message: Callable            # RuleContext -> str
    citation: str = ""

    def evaluate(self, ctx: RuleContext) -> Optional[ValidityFinding]:
        try:
            if not self.applies(ctx):
                return None
        except Exception:
            # A rule that cannot decide must never take down a valid fit.
            return None
        return ValidityFinding(self.id, self.severity, self.message(ctx),
                               self.citation)


def run_rules(rules: Sequence[ValidityRule], ctx: RuleContext) -> list:
    """Evaluate rules in order, dropping those that do not apply."""
    findings = []
    for rule in rules:
        finding = rule.evaluate(ctx)
        if finding is not None:
            findings.append(finding)
    return findings


SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def sort_findings(findings: Sequence[ValidityFinding]) -> list:
    """Most serious first, stable within a severity."""
    return sorted(findings, key=lambda f: SEVERITY_ORDER.get(f.severity, 99))


# ── Shared rule library ───────────────────────────────────────────────────────
# Each is a factory so a method can tune the threshold without copying the rule.

def poor_fit_rule(threshold: float = 0.95,
                  severity: str = "warning") -> ValidityRule:
    """The model may not describe this dataset."""
    return ValidityRule(
        id="poor_fit",
        severity=severity,
        applies=lambda c: c.gof.r_squared < threshold,
        message=lambda c: (
            f"Poor fit (R² = {c.gof.r_squared:.3f} < {threshold}) — the model may "
            "not fully describe this dataset. Possible causes: aquifer "
            "heterogeneity, boundary effects, or leakage."
        ),
    )


def confined_storativity_rule(low: float = 1e-6,
                              high: float = 1e-3) -> ValidityRule:
    """Storativity outside the range compaction-based storage can produce.

    Theis (1935) explicitly separates artesian storage (10^-5 to 10^-3) from
    gravity drainage in an unconfined aquifer (0.01 to 0.35). A value above the
    artesian range usually means the aquifer is not actually confined.
    """
    return ValidityRule(
        id="storativity_out_of_range",
        severity="warning",
        applies=lambda c: not (low <= c.value("S", 1e-4) <= high),
        message=lambda c: (
            f"S = {c.value('S'):.2e} is "
            f"{'above' if c.value('S') > high else 'below'} the typical confined "
            f"aquifer range ({low:.0e} – {high:.0e}). "
            + ("Verify the aquifer is truly artesian; higher S suggests unconfined "
               "or leaky conditions." if c.value("S") > high else
               "Check for leakage or partial penetration effects.")
        ),
        citation="Theis (1935)",
    )


def bound_hit_rule() -> ValidityRule:
    """A parameter pinned to a bound is a floor or ceiling, not a measurement."""

    def hits(c):
        return [n for n, side in c.bound_hits.items() if side]

    def text(c):
        parts = []
        for name in hits(c):
            side = c.bound_hits[name]
            spec = c.spec(name)
            limit = spec.lower if side == "lower" else spec.upper
            parts.append(f"{name} reached its {side} bound ({limit:g} {spec.unit})")
        return (
            "; ".join(parts)
            + ". The true value may lie beyond the range this method allows, so "
            "treat the estimate as a limit rather than a measurement."
        )

    return ValidityRule(
        id="parameter_at_bound",
        severity="warning",
        applies=lambda c: bool(hits(c)),
        message=text,
    )


def ci_unavailable_rule(message: str = "") -> ValidityRule:
    """No estimated parameter has a confidence interval.

    covariance_to_ci returns None for every parameter when the covariance
    diagonal holds inf or a negative value — the fit did not settle on a unique
    solution, or the Jacobian is ill-conditioned. Without this rule the interval
    simply disappears from the results and the user is never told why. `message`
    lets a method explain the likely cause in the terms of its own paper.
    """

    def applies(c):
        free = [name for name, spec in c.specs.items() if spec.is_free]
        return (bool(c.intervals) and bool(free)
                and all(c.intervals.get(name) is None for name in free))

    return ValidityRule(
        id="confidence_intervals_unavailable",
        severity="warning",
        applies=applies,
        message=lambda c: message or (
            "Confidence intervals could not be computed — the covariance matrix "
            "is ill-conditioned, so the data do not constrain every parameter. "
            "Treat the estimates as indicative."
        ),
    )


def parameter_correlation_rule(threshold: float = 0.98) -> ValidityRule:
    """Two parameters trading off almost perfectly — a non-unique fit.

    This is the dominant failure mode of four- and six-parameter unconfined and
    fractured fits, and it is invisible in the usual diagnostics: the optimiser
    converges, the covariance is finite, and R-squared is excellent. Only the
    off-diagonal correlation reveals that the individual values are arbitrary
    along a valley in parameter space.
    """

    def offenders(c):
        return [(pair, rho) for pair, rho in c.correlations.items()
                if abs(rho) >= threshold]

    def text(c):
        listed = ", ".join(
            f"{a} and {b} (r = {rho:+.3f})" for (a, b), rho in offenders(c)
        )
        return (
            f"Parameters are almost perfectly correlated: {listed}. The fit is "
            "not unique — many combinations of these values reproduce the data "
            "equally well, so the individual estimates are unreliable even though "
            "the curve matches. Fix one parameter from independent knowledge, or "
            "use a simpler model."
        )

    return ValidityRule(
        id="parameters_correlated",
        severity="critical",
        applies=lambda c: bool(offenders(c)),
        message=text,
    )


def plausible_range_rule(param: str) -> ValidityRule:
    """Flag a parameter outside the range declared on its ParameterSpec."""

    def out_of_range(c):
        spec = c.spec(param)
        if spec is None or spec.plausible_range is None:
            return False
        low, high = spec.plausible_range
        return not (low <= c.value(param) <= high)

    def text(c):
        spec = c.spec(param)
        low, high = spec.plausible_range
        return (
            f"{param} = {c.value(param):.3e} {spec.unit} falls outside the "
            f"physically typical range ({low:.1e} – {high:.1e} {spec.unit}). "
            "This can be genuine, but check the input units and the test setup."
        )

    return ValidityRule(
        id=f"implausible_{param}",
        severity="warning",
        applies=out_of_range,
        message=text,
    )


def overfitting_rule(min_points_per_param: int = 8) -> ValidityRule:
    """Too many parameters for the amount of data present.

    R-squared cannot detect this — it improves monotonically with parameter
    count. This rule is the reason GoodnessOfFit carries AICc.
    """
    return ValidityRule(
        id="too_few_points_per_parameter",
        severity="warning",
        applies=lambda c: (c.gof.n_params > 0
                           and c.gof.n_points / c.gof.n_params < min_points_per_param),
        message=lambda c: (
            f"Only {c.gof.n_points} points for {c.gof.n_params} fitted parameters "
            f"({c.gof.n_points / c.gof.n_params:.1f} per parameter). R² will look "
            "good regardless; compare models on AICc rather than R², and prefer a "
            "simpler model unless the extra parameters are clearly justified."
        ),
    )


def cross_method_hint_rule(other_label: str,
                           reason: str = "") -> ValidityRule:
    """Always-on suggestion to corroborate with an independent method."""
    return ValidityRule(
        id="compare_with_other_method",
        severity="info",
        applies=lambda c: True,
        message=lambda c: (
            f"{other_label} provides an independent estimate on the same data. "
            + (reason or "Comparing both methods is good practice.")
        ),
    )
