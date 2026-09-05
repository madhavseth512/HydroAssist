"""Tests for FittedParameter and the generalised CalculationResult.

This is the compatibility keystone of the architecture migration. The whole point
is that two kinds of writer coexist:

  * the three shipped calculators, which pass `T=` and `S=` and know nothing
    about parameter lists
  * methods added later, which build a `parameters` list of four or six entries
    and never mention T or S directly

and that ~30 existing read sites — `result.T`, `result.S`, `result.T_day`,
`result.T_ci` in plots.py, report.py and app.py — keep working under both. The
tests below pin both directions.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.calculator.base import CalculationInput, CalculationResult
from src.calculator.fitting import GoodnessOfFit
from src.calculator.parameters import (
    FittedParameter, storativity, transmissivity,
)
from src.calculator.rules import ValidityFinding
from src.calculator.theis import TheisCalculator


# ── FittedParameter ───────────────────────────────────────────────────────────

class TestFittedParameter:

    def test_symbol_defaults_to_the_name(self):
        assert FittedParameter("Sy", 0.15, "-").symbol == "Sy"

    def test_display_unit_defaults_to_the_si_unit(self):
        assert FittedParameter("S", 5e-5, "-").display_unit == "-"

    def test_display_applies_the_scale(self):
        p = transmissivity(1e-3)
        assert p.display == pytest.approx(86.4)
        assert p.display_unit == "m^2/day"

    def test_display_ci_applies_the_scale(self):
        p = transmissivity(1e-3, ci=(9e-4, 1.1e-3))
        lo, hi = p.display_ci
        assert lo == pytest.approx(77.76)
        assert hi == pytest.approx(95.04)

    def test_display_ci_is_none_without_an_interval(self):
        assert transmissivity(1e-3).display_ci is None

    def test_estimated_is_false_for_fixed_and_derived(self):
        assert FittedParameter("T", 1e-3, "m^2/s").estimated is True
        assert FittedParameter("T", 1e-3, "m^2/s", fixed=True).estimated is False
        assert FittedParameter("T", 1e-3, "m^2/s", derived=True).estimated is False

    def test_round_trips_through_a_dict(self):
        original = FittedParameter(
            "beta", 0.05, "-", symbol="β", ci=(0.03, 0.08),
            reliable=False, reliability_note="strongly correlated with Sy",
            at_bound="upper",
        )
        restored = FittedParameter.from_dict(original.to_dict())

        assert restored == original

    def test_to_dict_is_json_safe(self):
        payload = transmissivity(1e-3, ci=(9e-4, 1.1e-3)).to_dict()
        json.dumps(payload)
        assert isinstance(payload["ci"], list)


# ── Legacy writer: T= / S= ────────────────────────────────────────────────────

class TestLegacyWriterStillWorks:
    """The three shipped calculators construct results this way."""

    def test_t_and_s_construct_as_before(self):
        r = CalculationResult(method="Theis (1935)", T=2.5e-3, S=5e-5)

        assert r.T == 2.5e-3
        assert r.S == 5e-5
        assert r.T_day == pytest.approx(216.0)

    def test_parameters_are_synthesised_from_t_and_s(self):
        r = CalculationResult(method="Theis (1935)", T=2.5e-3, S=5e-5)

        assert r.parameter_names == ["T", "S"]
        assert r.value("T") == 2.5e-3
        assert r.get("S").unit == "-"

    def test_confidence_intervals_are_carried_onto_the_parameters(self):
        r = CalculationResult(method="Theis (1935)", T=2.5e-3, S=5e-5,
                              T_ci=(2.4e-3, 2.6e-3), S_ci=(4e-5, 6e-5))

        assert r.get("T").ci == (2.4e-3, 2.6e-3)
        assert r.get("S").ci == (4e-5, 6e-5)

    def test_failed_result_synthesises_no_parameters(self):
        r = CalculationResult(method="Theis (1935)", T=0.0, S=0.0,
                              success=False, error_message="did not converge")
        assert r.parameters == []

    def test_synthesised_transmissivity_displays_in_m2_per_day(self):
        r = CalculationResult(method="Theis (1935)", T=1e-3, S=5e-5)
        assert r.get("T").display == pytest.approx(86.4)


# ── New writer: parameters= ───────────────────────────────────────────────────

class TestNewWriterProjectsOntoLegacyFields:
    """A four- or six-parameter method writes `parameters` and never sets T or S.
    Every existing read site must still work."""

    def _neuman_like(self):
        return CalculationResult(
            method="Neuman (1974)",
            method_id="neuman_1974",
            parameters=[
                transmissivity(1.2e-3, ci=(1.1e-3, 1.3e-3)),
                storativity(4.0e-5, ci=(3e-5, 5e-5)),
                FittedParameter("Sy", 0.18, "-", symbol="S_y"),
                FittedParameter("beta", 0.05, "-", symbol="β"),
            ],
        )

    def test_t_and_s_are_projected_back(self):
        r = self._neuman_like()

        assert r.T == pytest.approx(1.2e-3)
        assert r.S == pytest.approx(4.0e-5)

    def test_t_day_is_derived_from_the_projection(self):
        assert self._neuman_like().T_day == pytest.approx(103.68)

    def test_confidence_intervals_are_projected_back(self):
        r = self._neuman_like()

        assert r.T_ci == (1.1e-3, 1.3e-3)
        assert r.S_ci == (3e-5, 5e-5)

    def test_extra_parameters_are_reachable(self):
        r = self._neuman_like()

        assert r.value("Sy") == pytest.approx(0.18)
        assert r.parameter_names == ["T", "S", "Sy", "beta"]

    def test_a_method_without_t_leaves_the_legacy_field_alone(self):
        """Warren-Root fits omega and lambda; T is simply absent."""
        r = CalculationResult(
            method="Warren-Root (1963)",
            parameters=[FittedParameter("omega", 0.1, "-"),
                        FittedParameter("lambda", 1e-6, "-")],
        )
        assert r.T == 0.0
        assert r.value("omega") == 0.1

    def test_derived_transmissivity_is_flagged(self):
        """Moench fits K and Ss; T = K*b is worth showing but is not fitted."""
        r = CalculationResult(
            method="Moench (1984)",
            parameters=[
                FittedParameter("K", 1e-5, "m/s"),
                FittedParameter("Ss", 1e-6, "1/m"),
                transmissivity(2e-4, derived=True),
            ],
        )
        assert r.T == pytest.approx(2e-4)
        assert r.get("T").derived is True
        assert r.get("T").estimated is False

    def test_unreliable_parameters_are_listed(self):
        """Papadopulos & Cooper (1967) p. 244 on storativity."""
        r = CalculationResult(
            method="Papadopulos-Cooper (1967)",
            parameters=[
                transmissivity(1e-3),
                storativity(1e-4, reliable=False,
                            reliability_note="order-of-magnitude only (paper p. 244)"),
            ],
        )
        assert [p.name for p in r.unreliable] == ["S"]

    def test_parameters_at_bounds_are_listed(self):
        r = CalculationResult(
            method="Theis (1935)",
            parameters=[transmissivity(1.0, at_bound="upper"), storativity(5e-5)],
        )
        assert [p.name for p in r.at_bounds] == ["T"]


# ── Goodness of fit and findings ──────────────────────────────────────────────

class TestGofAndFindings:

    def test_gof_mirrors_onto_the_legacy_scalars(self):
        gof = GoodnessOfFit(rmse=0.0024, r_squared=0.9999,
                            adjusted_r_squared=0.9998, ss_res=1.0, ss_tot=1e4,
                            n_points=60, n_params=2, aic=-100.0, aicc=-99.0)
        r = CalculationResult(method="Theis (1935)", T=2.5e-3, S=5e-5, gof=gof)

        assert r.rmse == pytest.approx(0.0024)
        assert r.r_squared == pytest.approx(0.9999)

    def test_findings_render_down_to_validity_notes(self):
        r = CalculationResult(
            method="Theis (1935)", T=2.5e-3, S=5e-5,
            findings=[ValidityFinding("poor_fit", "warning", "R² is low"),
                      ValidityFinding("compare", "info", "Try Cooper-Jacob")],
        )
        assert r.validity_notes == ["R² is low", "Try Cooper-Jacob"]

    def test_explicit_notes_are_not_overwritten_by_findings(self):
        r = CalculationResult(
            method="Theis (1935)", T=2.5e-3, S=5e-5,
            validity_notes=["hand written"],
            findings=[ValidityFinding("poor_fit", "warning", "R² is low")],
        )
        assert r.validity_notes == ["hand written"]


# ── Serialisation ─────────────────────────────────────────────────────────────

class TestStateDict:

    def test_every_legacy_key_is_preserved(self):
        r = CalculationResult(method="Theis (1935)", T=2.5e-3, S=5e-5,
                              T_ci=(2.4e-3, 2.6e-3), rmse=0.002, r_squared=0.999)
        payload = r.to_state_dict()

        for key in ["method", "T", "S", "T_day", "T_ci", "S_ci", "rmse",
                    "r_squared", "validity_notes", "success", "error_message"]:
            assert key in payload, f"{key} must not disappear from AgentState"

    def test_new_keys_are_additive_and_json_safe(self):
        r = CalculationResult(
            method="Neuman (1974)", method_id="neuman_1974",
            parameters=[transmissivity(1.2e-3), FittedParameter("Sy", 0.18, "-")],
            diagnostics={"beta": 0.05, "n_excluded": 3},
        )
        payload = r.to_state_dict()
        json.dumps(payload)

        assert payload["method_id"] == "neuman_1974"
        assert [p["name"] for p in payload["parameters"]] == ["T", "Sy"]
        assert payload["diagnostics"]["beta"] == pytest.approx(0.05)

    def test_non_numeric_diagnostics_are_dropped_to_stay_serialisable(self):
        r = CalculationResult(
            method="Theis (1935)", T=2.5e-3, S=5e-5,
            diagnostics={"u_max": 0.018, "note": "text", "arr": np.arange(3)},
        )
        payload = r.to_state_dict()
        json.dumps(payload)

        assert "u_max" in payload["diagnostics"]
        assert "note" not in payload["diagnostics"]
        assert "arr" not in payload["diagnostics"]

    def test_booleans_are_not_smuggled_in_as_numbers(self):
        r = CalculationResult(method="Theis (1935)", T=2.5e-3, S=5e-5,
                              diagnostics={"converged": True})
        assert "converged" not in r.to_state_dict()["diagnostics"]


# ── Against a real fit ────────────────────────────────────────────────────────

class TestAgainstARealCalculator:

    @pytest.fixture(scope="class")
    def result(self):
        df = pd.read_csv("tests/theis_cooper_jacob_sample.csv")
        return TheisCalculator().calculate(
            CalculationInput(df=df, Q=500 / 86400, r=50.0))

    def test_shipped_calculator_now_exposes_parameters(self, result):
        assert result.parameter_names == ["T", "S"]

    def test_parameter_values_agree_with_the_legacy_fields(self, result):
        assert result.value("T") == result.T
        assert result.value("S") == result.S

    def test_parameter_intervals_agree_with_the_legacy_fields(self, result):
        assert result.get("T").ci == result.T_ci
        assert result.get("S").ci == result.S_ci

    def test_state_dict_from_a_real_fit_is_serialisable(self, result):
        json.dumps(result.to_state_dict())
