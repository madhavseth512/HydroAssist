"""Unit tests for the shared input validation and the validity-rule library.

`validate_series` replaces three verbatim copies of the same checks. Its error
strings are asserted here because they are matched by existing calculator tests
and, more importantly, because they are what the user reads when their data is
rejected — each has to say what is wrong and what to do.

The rule library is where interpretation lives. Two rules carry real weight:
`parameter_correlation_rule` catches the non-unique fits that look perfect on
every other diagnostic, and `overfitting_rule` catches the case R-squared is
structurally unable to see.
"""

import numpy as np
import pandas as pd
import pytest

from src.calculator.base import CalculationInput
from src.calculator.fitting import GoodnessOfFit, ParameterSpec
from src.calculator.rules import (
    RuleContext, ValidityFinding, ValidityRule,
    bound_hit_rule, confined_storativity_rule, cross_method_hint_rule,
    overfitting_rule, parameter_correlation_rule, plausible_range_rule,
    poor_fit_rule, run_rules, sort_findings,
)
from src.calculator.validation import (
    MIN_DATA_POINTS, require_geometry, resolve_path, validate_series,
)


def _inputs(n=30, drawdown=None, times=None, **kw):
    t = times if times is not None else np.logspace(1, 4, n)
    s = drawdown if drawdown is not None else 0.15 * np.log10(t / 1.0)
    df = pd.DataFrame({"time_s": t, "drawdown_m": s})
    kw.setdefault("Q", 500 / 86400)
    kw.setdefault("r", 50.0)
    return CalculationInput(df=df, **kw)


# ── validate_series ───────────────────────────────────────────────────────────

class TestValidateSeries:

    def test_good_data_passes(self):
        assert validate_series(_inputs()) is None

    def test_too_few_points_is_rejected(self):
        err = validate_series(_inputs(n=MIN_DATA_POINTS - 1))
        assert err is not None and "Insufficient data" in err

    def test_exactly_the_minimum_passes(self):
        assert validate_series(_inputs(n=MIN_DATA_POINTS)) is None

    def test_negative_drawdown_names_the_likely_cause(self):
        t = np.logspace(1, 4, 30)
        s = 0.15 * np.log10(t)
        s[5] = -0.2
        err = validate_series(_inputs(times=t, drawdown=s))

        assert "Negative drawdown" in err
        assert "recovery" in err, "the message should say what to do about it"

    def test_non_positive_time_is_rejected(self):
        # Drawdown supplied explicitly so this isolates the time check — deriving
        # it from t would make the t=0 row -inf and trip the drawdown check first.
        t = np.logspace(1, 4, 30)
        t[0] = 0.0
        err = validate_series(_inputs(times=t, drawdown=np.linspace(0.1, 2.0, 30)))

        assert "Non-positive time" in err
        assert "formatter" in err, "should point at where t=0 gets removed"

    def test_message_wording_matches_the_shipped_calculators(self):
        """Existing calculator tests match on these strings."""
        err = validate_series(_inputs(n=5))
        assert err.startswith("Insufficient data:")
        assert f"minimum {MIN_DATA_POINTS} required" in err

    def test_small_radius_warns_but_does_not_reject(self, caplog):
        """A piezometer close to the pumped well is unusual, not impossible."""
        with caplog.at_level("WARNING"):
            assert validate_series(_inputs(r=0.2)) is None
        assert any("very small" in r.message for r in caplog.records)

    def test_rate_can_be_waived_for_test_types_without_one(self):
        """A slug test has no pumping rate; the shared checks must still apply."""
        t = np.logspace(1, 3, 20)
        df = pd.DataFrame({"time_s": t, "drawdown_m": np.linspace(1.0, 0.01, 20)})
        inputs = CalculationInput(df=df, Q=0.001, r=50.0)

        assert validate_series(inputs, require_rate=False,
                               require_distance=False) is None

    def test_custom_minimum_is_honoured(self):
        assert validate_series(_inputs(n=6), min_points=5) is None
        assert validate_series(_inputs(n=6), min_points=20) is not None


# ── Geometry requirements ─────────────────────────────────────────────────────

class TestResolvePath:

    def test_reads_a_plain_attribute(self):
        assert resolve_path(_inputs(), "Q") == pytest.approx(500 / 86400)

    def test_missing_attribute_is_none_not_an_error(self):
        assert resolve_path(_inputs(), "nonexistent") is None

    def test_missing_nested_attribute_is_none(self):
        assert resolve_path(_inputs(), "aquifer.b") is None

    def test_indexed_path_on_a_missing_collection_is_none(self):
        assert resolve_path(_inputs(), "aquitards[0].b_prime") is None


class TestRequireGeometry:

    def test_satisfied_requirement_returns_none(self):
        assert require_geometry(_inputs(), ("Q", "r")) is None

    def test_no_requirements_returns_none(self):
        assert require_geometry(_inputs(), ()) is None

    def test_missing_requirement_is_reported_in_hydrogeology_terms(self):
        err = require_geometry(_inputs(), ("aquifer.b",), "Neuman (1974)")

        assert "Neuman (1974)" in err
        assert "saturated aquifer thickness" in err, "not the dotted path"
        assert "Step 2" in err, "the message should say where to fix it"

    def test_several_missing_requirements_are_listed_together(self):
        err = require_geometry(_inputs(),
                               ("aquifer.b", "aquifer.Kz_over_Kr"), "Neuman (1974)")
        assert "thickness" in err and "anisotropy" in err
        assert "were not supplied" in err

    def test_unknown_path_falls_back_to_the_raw_name(self):
        err = require_geometry(_inputs(), ("some.new.path",))
        assert "some.new.path" in err


# ── Rule mechanics ────────────────────────────────────────────────────────────

def _ctx(params=None, r_squared=0.999, n_points=60, n_params=2,
         bound_hits=None, correlations=None, specs=None):
    t = np.logspace(1, 4, n_points)
    return RuleContext(
        params=params or {"T": 2.5e-3, "S": 5e-5},
        specs=specs or {
            "T": ParameterSpec("T", "m^2/s", 1e-9, 1.0, 1e-3),
            "S": ParameterSpec("S", "-", 1e-9, 1e-2, 1e-4),
        },
        gof=GoodnessOfFit(rmse=0.002, r_squared=r_squared,
                          adjusted_r_squared=r_squared, ss_res=1.0, ss_tot=100.0,
                          n_points=n_points, n_params=n_params,
                          aic=-100.0, aicc=-99.0),
        time_s=t, drawdown_obs=0.1 * np.log10(t), drawdown_fitted=0.1 * np.log10(t),
        bound_hits=bound_hits or {},
        correlations=correlations or {},
    )


class TestRuleMechanics:

    def test_rule_that_does_not_apply_produces_nothing(self):
        assert poor_fit_rule().evaluate(_ctx(r_squared=0.999)) is None

    def test_rule_that_applies_produces_a_finding(self):
        finding = poor_fit_rule().evaluate(_ctx(r_squared=0.5))

        assert isinstance(finding, ValidityFinding)
        assert finding.id == "poor_fit"
        assert finding.severity == "warning"
        assert "0.500" in finding.text

    def test_a_raising_rule_never_breaks_a_valid_fit(self):
        """An interpretation bug must not destroy a correct numerical result."""
        exploding = ValidityRule(
            id="boom", severity="info",
            applies=lambda c: 1 / 0,
            message=lambda c: "unreachable",
        )
        assert exploding.evaluate(_ctx()) is None

    def test_run_rules_keeps_order_and_drops_non_applying(self):
        findings = run_rules(
            [poor_fit_rule(), cross_method_hint_rule("Theis (1935)")],
            _ctx(r_squared=0.999),
        )
        assert [f.id for f in findings] == ["compare_with_other_method"]

    def test_sort_findings_puts_critical_first(self):
        findings = [
            ValidityFinding("a", "info", "i"),
            ValidityFinding("b", "critical", "c"),
            ValidityFinding("c", "warning", "w"),
        ]
        assert [f.severity for f in sort_findings(findings)] == [
            "critical", "warning", "info"]

    def test_finding_renders_as_its_text(self):
        assert str(ValidityFinding("x", "info", "hello")) == "hello"


# ── The shared rule library ───────────────────────────────────────────────────

class TestSharedRules:

    def test_poor_fit_threshold_is_configurable(self):
        assert poor_fit_rule(threshold=0.99).evaluate(_ctx(r_squared=0.97))
        assert poor_fit_rule(threshold=0.95).evaluate(_ctx(r_squared=0.97)) is None

    @pytest.mark.parametrize("S,expect_flag", [
        (5e-5, False),      # squarely confined
        (1e-3, False),      # top of the range, inclusive
        (5e-2, True),       # unconfined territory
        (1e-8, True),       # implausibly low
    ])
    def test_confined_storativity_range(self, S, expect_flag):
        finding = confined_storativity_rule().evaluate(_ctx(params={"T": 2.5e-3, "S": S}))
        assert (finding is not None) == expect_flag

    def test_high_storativity_message_suggests_the_right_cause(self):
        finding = confined_storativity_rule().evaluate(
            _ctx(params={"T": 2.5e-3, "S": 5e-2}))
        assert "unconfined" in finding.text

    def test_low_storativity_message_suggests_the_right_cause(self):
        finding = confined_storativity_rule().evaluate(
            _ctx(params={"T": 2.5e-3, "S": 1e-8}))
        assert "leakage" in finding.text

    def test_bound_hit_is_reported_as_a_limit_not_a_measurement(self):
        finding = bound_hit_rule().evaluate(_ctx(bound_hits={"T": "upper", "S": None}))

        assert finding is not None
        assert "upper bound" in finding.text
        assert "limit rather than a measurement" in finding.text

    def test_no_bound_hits_produces_nothing(self):
        assert bound_hit_rule().evaluate(_ctx(bound_hits={"T": None})) is None

    def test_plausible_range_rule_uses_the_spec(self):
        specs = {"T": ParameterSpec("T", "m^2/s", 1e-9, 1.0, 1e-3,
                                    plausible_range=(1e-5, 1e-2))}
        assert plausible_range_rule("T").evaluate(
            _ctx(params={"T": 0.5}, specs=specs)) is not None
        assert plausible_range_rule("T").evaluate(
            _ctx(params={"T": 1e-3}, specs=specs)) is None

    def test_plausible_range_rule_is_silent_without_a_declared_range(self):
        assert plausible_range_rule("T").evaluate(_ctx()) is None

    def test_cross_method_hint_always_applies(self):
        finding = cross_method_hint_rule("Theis (1935)").evaluate(_ctx())
        assert finding.severity == "info"
        assert "Theis (1935)" in finding.text


class TestCorrelationRule:
    """The failure mode that every other diagnostic misses."""

    def test_flags_near_perfect_correlation_as_critical(self):
        finding = parameter_correlation_rule().evaluate(
            _ctx(correlations={("Sy", "beta"): 0.995}))

        assert finding is not None
        assert finding.severity == "critical"
        assert "not unique" in finding.text

    def test_message_says_what_to_do(self):
        finding = parameter_correlation_rule().evaluate(
            _ctx(correlations={("Sy", "beta"): 0.995}))
        assert "Fix one parameter" in finding.text or "simpler model" in finding.text

    def test_negative_correlation_counts_too(self):
        assert parameter_correlation_rule().evaluate(
            _ctx(correlations={("T", "S"): -0.99})) is not None

    def test_moderate_correlation_is_not_flagged(self):
        """Some correlation between T and S is normal and expected."""
        assert parameter_correlation_rule().evaluate(
            _ctx(correlations={("T", "S"): 0.7})) is None

    def test_threshold_is_configurable(self):
        ctx = _ctx(correlations={("T", "S"): 0.95})
        assert parameter_correlation_rule(threshold=0.9).evaluate(ctx) is not None
        assert parameter_correlation_rule(threshold=0.99).evaluate(ctx) is None


class TestOverfittingRule:

    def test_flags_too_few_points_per_parameter(self):
        finding = overfitting_rule().evaluate(_ctx(n_points=20, n_params=6))

        assert finding is not None
        assert "AICc" in finding.text, "should point at the right statistic"

    def test_comfortable_ratio_is_not_flagged(self):
        assert overfitting_rule().evaluate(_ctx(n_points=60, n_params=2)) is None

    def test_threshold_is_configurable(self):
        ctx = _ctx(n_points=30, n_params=3)   # 10 per parameter
        assert overfitting_rule(min_points_per_param=15).evaluate(ctx) is not None
        assert overfitting_rule(min_points_per_param=8).evaluate(ctx) is None
