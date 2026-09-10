"""Tests for projects — saved, reopenable units of work.

Three things are pinned here.

1. **A project survives the round trip.** Save to JSON, reopen, and every field —
   metadata, readings, setup, analyses with their parameters and CIs — comes back
   identical. Including the awkward values: NaN and infinity have no JSON form and
   must not break a save.

2. **Bad files fail with a message the user can act on**, not a traceback: wrong
   file, damaged file, newer format, oversized upload.

3. **Model comparison is meaningful.** The decisive tests fit real calculators to
   synthetic data of known type and check the AICc ranking picks the true model:
   on leaky data Hantush-Jacob must beat Theis, and on plain Theis data the extra
   leakage parameter must NOT win. Ranking by R² would fail the second — which is
   exactly why the comparison never uses it.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.special import expn

from src.calculator.base import CalculationInput
from src.calculator.methods.hantush_jacob import (
    HantushJacobCalculator, hantush_jacob_drawdown,
)
from src.calculator.theis import TheisCalculator
from src.project.model import (
    FILE_SUFFIX, MAX_FILE_BYTES, SCHEMA_VERSION, AnalysisRecord, Project,
    ProjectFileError, ProjectInfo, Readings, TestSetup,
)

Q, R = 500 / 86400, 50.0


def _theis_frame(noise=0.002, seed=1):
    t = np.logspace(1, 5, 60)
    u = (R ** 2 * 5e-5) / (4.0 * 2.5e-3 * t)
    s = (Q / (4.0 * np.pi * 2.5e-3)) * expn(1, np.maximum(u, 1e-12))
    s = s + np.random.default_rng(seed).normal(0, noise, t.size)
    return pd.DataFrame({"time_s": t, "drawdown_m": s})


def _leaky_frame(noise=0.002, seed=2):
    t = np.logspace(1, 6, 70)
    s = hantush_jacob_drawdown(t, 2.5e-3, 5e-5, 300.0, Q, R)
    s = s + np.random.default_rng(seed).normal(0, noise, t.size)
    return pd.DataFrame({"time_s": t, "drawdown_m": s})


def _fit(calculator, df):
    return calculator().calculate(CalculationInput(df=df, Q=Q, r=R))


def _project_with_one_analysis():
    project = Project(info=ProjectInfo(title="Kharagpur campus well field",
                                       client="IIT Kharagpur",
                                       project_number="BTP-2026-04",
                                       location="Kharagpur, West Bengal",
                                       test_date="2026-09-01",
                                       analyst="M. Seth"))
    test = project.add_test("PW-1 constant rate")
    test.pumping_well, test.observation_well = "PW-1", "OW-2"
    df = _theis_frame()
    test.set_readings(Readings.from_frame(df, source_name="pw1.csv",
                                          conversions=["Time: minutes → seconds"]))
    test.setup = TestSetup(Q_m3s=Q, Q_entered=500.0, Q_unit="m³/day", r_m=R)
    test.add_analysis(_fit(TheisCalculator, df), "Theis (1935)", "theis_1935")
    return project, test


# ── Round trip ────────────────────────────────────────────────────────────────

class TestRoundTrip:

    def test_everything_survives_save_and_reopen(self):
        project, _ = _project_with_one_analysis()
        reopened = Project.from_json(project.to_json())

        assert reopened.to_dict() == project.to_dict()

    def test_metadata_is_preserved(self):
        project, _ = _project_with_one_analysis()
        info = Project.from_json(project.to_json()).info

        assert info.client == "IIT Kharagpur"
        assert info.location == "Kharagpur, West Bengal"
        assert info.test_date == "2026-09-01"

    def test_readings_come_back_as_the_same_frame(self):
        project, test = _project_with_one_analysis()
        again = Project.from_json(project.to_json()).get_test(test.id)

        pd.testing.assert_frame_equal(again.readings.to_frame(),
                                      test.readings.to_frame())

    def test_the_entered_q_unit_is_preserved_for_display(self):
        project, test = _project_with_one_analysis()
        setup = Project.from_json(project.to_json()).get_test(test.id).setup

        assert setup.Q_entered == 500.0 and setup.Q_unit == "m³/day"
        assert setup.Q_m3s == pytest.approx(Q)

    def test_fitted_parameters_and_intervals_are_preserved(self):
        project, test = _project_with_one_analysis()
        original = test.analyses[0].result
        again = Project.from_json(project.to_json()).get_test(test.id).analyses[0].result

        assert again["T"] == pytest.approx(original["T"])
        assert again["T_ci"] == pytest.approx(original["T_ci"])
        assert [p["name"] for p in again["parameters"]] == ["T", "S"]

    def test_the_file_is_plain_readable_json(self):
        project, _ = _project_with_one_analysis()
        data = json.loads(project.to_json())

        assert data["schema_version"] == SCHEMA_VERSION
        assert data["info"]["title"] == "Kharagpur campus well field"

    def test_non_finite_values_do_not_break_a_save(self):
        """JSON has no NaN or infinity. AICc is infinite when parameters exhaust
        the data, so this happens in practice."""
        project, test = _project_with_one_analysis()
        test.analyses[0].result["gof"]["aicc"] = float("inf")
        test.analyses[0].result["weird"] = [float("nan"), 1.0]

        data = json.loads(project.to_json())   # allow_nan=False would raise
        analysis = data["tests"][0]["analyses"][0]["result"]
        assert analysis["gof"]["aicc"] is None
        assert analysis["weird"] == [None, 1.0]

    def test_multi_well_readings_round_trip(self):
        df = pd.DataFrame({"time_s": [1.0, 2.0, 1.0, 2.0],
                           "drawdown_m": [0.1, 0.2, 0.05, 0.1],
                           "well_id": ["OW-1", "OW-1", "OW-2", "OW-2"]})
        project = Project()
        project.add_test().set_readings(Readings.from_frame(df))

        again = Project.from_json(project.to_json())
        pd.testing.assert_frame_equal(again.tests[0].readings.to_frame(), df)

    def test_filename_is_safe_and_carries_the_suffix(self):
        project = Project(info=ProjectInfo(title="Site 4 / well: A?"))
        assert project.filename.endswith(FILE_SUFFIX)
        assert "/" not in project.filename and ":" not in project.filename


# ── Bad files ─────────────────────────────────────────────────────────────────

class TestBadFilesFailClearly:

    def test_non_json_is_rejected(self):
        with pytest.raises(ProjectFileError, match="not valid JSON"):
            Project.from_json(b"time_s,drawdown_m\n1,0.1\n")

    def test_someone_elses_json_is_rejected(self):
        with pytest.raises(ProjectFileError, match="not a HydroAssist project"):
            Project.from_json(json.dumps({"hello": "world"}).encode())

    def test_a_newer_format_says_to_update(self):
        data = Project().to_dict()
        data["schema_version"] = SCHEMA_VERSION + 1

        with pytest.raises(ProjectFileError, match="newer version"):
            Project.from_dict(data)

    def test_a_damaged_file_says_it_is_damaged(self):
        project, _ = _project_with_one_analysis()
        data = project.to_dict()
        del data["tests"][0]["analyses"][0]["method_label"]

        with pytest.raises(ProjectFileError, match="damaged or incomplete"):
            Project.from_dict(data)

    def test_oversized_uploads_are_refused_before_parsing(self):
        with pytest.raises(ProjectFileError, match="limited"):
            Project.from_json(b" " * (MAX_FILE_BYTES + 1))

    def test_mismatched_readings_are_rejected(self):
        with pytest.raises(ValueError, match="same number"):
            Readings(time_s=[1.0, 2.0], drawdown_m=[0.1])


# ── Tests within a project ────────────────────────────────────────────────────

class TestProjectStructure:

    def test_tests_are_named_in_sequence_by_default(self):
        project = Project()
        assert [project.add_test().name for _ in range(3)] == [
            "Test 1", "Test 2", "Test 3"]

    def test_get_and_remove_test(self):
        project = Project()
        keep, drop = project.add_test("keep"), project.add_test("drop")
        project.remove_test(drop.id)

        assert project.get_test(keep.id).name == "keep"
        with pytest.raises(KeyError):
            project.get_test(drop.id)

    def test_an_analysis_needs_readings_first(self):
        test = Project().add_test()
        with pytest.raises(ValueError, match="Load readings"):
            test.add_analysis({"success": True}, "Theis (1935)")

    def test_analysis_count_spans_all_tests(self):
        project, _ = _project_with_one_analysis()
        second = project.add_test()
        df = _theis_frame(seed=9)
        second.set_readings(Readings.from_frame(df))
        second.add_analysis(_fit(TheisCalculator, df), "Theis (1935)")

        assert project.analysis_count == 2

    def test_the_setup_is_snapshotted_not_shared(self):
        """Changing Q later must not rewrite what an earlier analysis used."""
        project, test = _project_with_one_analysis()
        test.setup.Q_m3s = 999.0

        assert test.analyses[0].setup.Q_m3s == pytest.approx(Q)


# ── Staleness ─────────────────────────────────────────────────────────────────

class TestAnalysesOnDifferentDataAreNeverCompared:

    def test_replacing_readings_marks_earlier_analyses_stale(self):
        project, test = _project_with_one_analysis()
        test.set_readings(Readings.from_frame(_theis_frame(seed=42)))

        assert test.is_stale(test.analyses[0])

    def test_stale_analyses_are_kept_but_left_out_of_the_comparison(self):
        project, test = _project_with_one_analysis()
        test.set_readings(Readings.from_frame(_theis_frame(seed=42)))

        assert len(test.analyses) == 1
        assert test.comparison() == []

    def test_identical_readings_share_a_fingerprint(self):
        df = _theis_frame()
        assert Readings.from_frame(df).fingerprint == Readings.from_frame(df).fingerprint

    def test_a_single_changed_value_changes_the_fingerprint(self):
        df = _theis_frame()
        altered = df.copy()
        altered.loc[5, "drawdown_m"] += 1e-6

        assert (Readings.from_frame(df).fingerprint
                != Readings.from_frame(altered).fingerprint)


# ── Model comparison ──────────────────────────────────────────────────────────

def _compare_on(df):
    project = Project()
    test = project.add_test()
    test.set_readings(Readings.from_frame(df))
    test.add_analysis(_fit(TheisCalculator, df), "Theis (1935)", "theis_1935")
    test.add_analysis(_fit(HantushJacobCalculator, df), "Hantush-Jacob (1955)",
                      "hantush_jacob_1955")
    return test.comparison()


class TestModelComparisonPicksTheTrueModel:

    def test_on_leaky_data_hantush_jacob_ranks_first(self):
        rows = _compare_on(_leaky_frame())

        assert rows[0]["method"] == "Hantush-Jacob (1955)"
        assert rows[1]["delta_aicc"] > 10, "the evidence should be decisive"

    def test_on_theis_data_the_extra_parameter_does_not_win(self):
        """The case R² gets wrong: Hantush-Jacob, with one more parameter, fits
        Theis data at least as well by R² — but AICc charges for the parameter,
        so the simpler true model must not lose."""
        rows = _compare_on(_theis_frame())
        by_method = {r["method"]: r for r in rows}

        assert by_method["Hantush-Jacob (1955)"]["r_squared"] >= \
            by_method["Theis (1935)"]["r_squared"] - 1e-4, "R² cannot tell them apart"
        assert by_method["Theis (1935)"]["delta_aicc"] <= 2.0, (
            "Theis must be the best model or within the conventional ΔAICc < 2 "
            "band of indistinguishable models")

    def test_akaike_weights_sum_to_one(self):
        rows = _compare_on(_leaky_frame())
        assert sum(r["akaike_weight"] for r in rows) == pytest.approx(1.0)

    def test_the_best_model_has_zero_delta(self):
        rows = _compare_on(_leaky_frame())
        assert rows[0]["delta_aicc"] == pytest.approx(0.0)

    def test_parameter_counts_are_honest(self):
        """Derived r/B must not be counted as a free parameter."""
        by_method = {r["method"]: r for r in _compare_on(_leaky_frame())}

        assert by_method["Theis (1935)"]["n_params"] == 2
        assert by_method["Hantush-Jacob (1955)"]["n_params"] == 3

    def test_comparison_survives_the_round_trip(self):
        project = Project()
        test = project.add_test()
        df = _leaky_frame()
        test.set_readings(Readings.from_frame(df))
        test.add_analysis(_fit(TheisCalculator, df), "Theis (1935)")
        test.add_analysis(_fit(HantushJacobCalculator, df), "Hantush-Jacob (1955)")

        again = Project.from_json(project.to_json()).get_test(test.id)
        assert [r["method"] for r in again.comparison()] == \
            [r["method"] for r in test.comparison()]

    def test_failed_analyses_are_left_out(self):
        project, test = _project_with_one_analysis()
        test.add_analysis({"success": False, "error_message": "did not converge"},
                          "Theis (1935)")

        assert len(test.comparison()) == 1


class TestGofInTheStateDict:
    """The serialised result must carry what the comparison needs."""

    def test_original_calculators_now_report_aicc(self):
        result = _fit(TheisCalculator, _theis_frame()).to_state_dict()

        assert result["gof"]["aicc"] is not None
        assert result["gof"]["n_params"] == 2
        # On the curve-fit template since step 8, so Theis supplies its own gof;
        # a full-curve fit uses every point, so the basis covers the whole record.
        assert result["gof"]["basis"] == "fit"
        assert result["gof"]["n_points"] == len(_theis_frame())

    def test_straight_line_aicc_covers_the_whole_record(self):
        """Cooper-Jacob fits only late-time points; its comparison statistics are
        still computed over every point so its AICc is comparable with Theis's."""
        from src.calculator.cooper_jacob import CooperJacobCalculator

        result = _fit(CooperJacobCalculator, _theis_frame()).to_state_dict()

        assert result["gof"]["basis"] == "all_points"
        assert result["gof"]["n_points"] == len(_theis_frame())

    def test_template_calculators_report_their_own_gof(self):
        result = _fit(HantushJacobCalculator, _leaky_frame()).to_state_dict()

        assert result["gof"]["basis"] == "fit"
        assert result["gof"]["n_params"] == 3

    def test_a_failed_result_has_no_gof(self):
        from src.calculator.base import CalculationResult

        payload = CalculationResult(method="Theis (1935)", success=False).to_state_dict()
        assert payload["gof"] is None

    def test_the_state_dict_stays_json_serialisable(self):
        json.dumps(_fit(TheisCalculator, _theis_frame()).to_state_dict(),
                   allow_nan=False)
