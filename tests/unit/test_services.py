"""The interface-free application services.

These behaviours used to live inside `app.py`, reachable only by driving
Streamlit. They are the seam the HTTP API will sit on, so they are tested
directly here: which calculator gets built for a setup, what is kept beside a
result, how a result reaches a project and a PDF, and what an uploaded file
turns into.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.calculator import registry
from src.project.model import Project, ProjectInfo, TestSetup
from src.project.workspace import new_project
from src.services import analysis, data_intake

THEIS_SAMPLE = "tests/theis_cooper_jacob_sample.csv"
RECOVERY_SAMPLE = "tests/theis_recovery_sample.csv"
PC_SAMPLE = "tests/papadopulos_cooper_sample.csv"


def _setup(**kw):
    base = dict(Q_m3s=500 / 86400, Q_entered=500.0, Q_unit="m³/day", r_m=50.0)
    base.update(kw)
    return TestSetup(**base)


def _recovery_setup(**kw):
    return _setup(test_type=registry.RECOVERY, t_p_s=86400.0, **kw)


@pytest.fixture(scope="module")
def theis_df():
    return pd.read_csv(THEIS_SAMPLE)


@pytest.fixture(scope="module")
def theis_run(theis_df):
    setup = _setup()
    result = analysis.run_analysis(theis_df, "theis_1935", setup)
    return result, analysis.run_record(setup, "theis_1935", result,
                                       n_points=len(theis_df))


class TestNoUserInterface:

    def test_services_never_import_a_ui_framework(self):
        """The whole point of the package: it must stay usable from the API."""
        for path in Path("src/services").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert "import streamlit" not in source, path
            assert "st." not in source.replace("test.", ""), path


class TestBuildInputs:

    def test_observation_well_method_uses_the_distance(self, theis_df):
        spec = registry.require("theis_1935")
        inputs = analysis.build_inputs(theis_df, _setup(r_m=42.0), spec)

        assert inputs.r == 42.0
        assert inputs.test_type == registry.CONSTANT_RATE

    def test_pumped_well_method_uses_the_screen_radius(self):
        """Papadopulos-Cooper is fitted at the well face, so r is r_w."""
        spec = registry.require("papadopulos_cooper_1967")
        df = pd.read_csv(PC_SAMPLE)
        inputs = analysis.build_inputs(
            df, _setup(Q_m3s=300 / 86400, r_m=50.0, r_w_m=0.30, r_c_m=0.20), spec)

        assert inputs.r == 0.30 and inputs.r_w == 0.30 and inputs.r_c == 0.20

    def test_recovery_builds_a_schedule_that_ends_at_zero(self):
        spec = registry.require("theis_recovery_1935")
        df = pd.read_csv(RECOVERY_SAMPLE)
        inputs = analysis.build_inputs(df, _recovery_setup(), spec)

        assert [step.Q for step in inputs.schedule.steps] == [500 / 86400, 0.0]
        assert inputs.schedule.steps[1].t_start_s == 86400.0
        assert inputs.options["recovery_clock"] == "since_stop"

    def test_recovery_without_a_duration_says_so(self):
        spec = registry.require("theis_recovery_1935")
        df = pd.read_csv(RECOVERY_SAMPLE)

        with pytest.raises(ValueError, match="pumping duration"):
            analysis.build_inputs(df, _setup(test_type=registry.RECOVERY), spec)


class TestRunAnalysis:

    def test_fits_the_bundled_sample(self, theis_run):
        result, _ = theis_run
        assert result.success
        assert result.T_day == pytest.approx(215.29, rel=1e-3)

    def test_accepts_an_id_or_a_label(self, theis_df):
        by_id = analysis.run_analysis(theis_df, "theis_1935", _setup())
        by_label = analysis.run_analysis(theis_df, "Theis (1935)", _setup())
        assert by_id.T == pytest.approx(by_label.T, rel=1e-12)

    def test_runs_a_recovery_analysis(self):
        df = pd.read_csv(RECOVERY_SAMPLE)
        result = analysis.run_analysis(df, "theis_recovery_1935", _recovery_setup())

        assert result.success, result.error_message
        assert result.T == pytest.approx(2.5e-3, rel=0.03)


class TestRunRecord:

    def test_records_the_setup_the_fit_used(self, theis_run):
        _, record = theis_run

        assert record["Q"] == pytest.approx(500 / 86400)
        assert record["Q_entered"] == 500.0 and record["Q_unit"] == "m³/day"
        assert record["Q_day"] == pytest.approx(500.0)
        assert record["method"] == "Theis (1935)"
        assert record["method_id"] == "theis_1935"
        assert record["n_points"] == 50

    def test_takes_the_excluded_count_from_the_calculator(self, theis_df):
        setup = _setup()
        result = analysis.run_analysis(theis_df, "cooper_jacob_1946", setup)
        record = analysis.run_record(setup, "cooper_jacob_1946", result)

        assert record["n_excluded"] == result.diagnostics["n_excluded"] > 0

    def test_a_failed_fit_excludes_nothing(self, theis_df):
        setup = _setup()
        result = analysis.run_analysis(theis_df.head(3), "theis_1935", setup)

        assert not result.success
        assert analysis.run_record(setup, "theis_1935", result)["n_excluded"] == 0

    def test_carries_the_recovery_fields(self):
        df = pd.read_csv(RECOVERY_SAMPLE)
        setup = _recovery_setup()
        result = analysis.run_analysis(df, "theis_recovery_1935", setup)
        record = analysis.run_record(setup, "theis_recovery_1935", result)

        assert record["test_type"] == registry.RECOVERY
        assert record["t_p_s"] == 86400.0
        assert record["recovery_clock"] == "since_stop"


class TestMethodChoice:

    def test_constant_rate_offers_every_constant_rate_method(self):
        offered, gated = analysis.method_options(None)

        assert offered == [s.label for s in registry.for_test_type(registry.CONSTANT_RATE)]
        assert gated == []

    def test_recovery_offers_only_recovery_methods(self):
        offered, _ = analysis.method_options(None, registry.RECOVERY)
        assert offered == ["Theis Recovery (1935)"]

    def test_the_validator_can_gate_a_method_out(self):
        class Validation:
            suitable_methods = ["Theis (1935)"]

        offered, gated = analysis.method_options(Validation())

        assert offered == ["Theis (1935)"]
        assert "Cooper-Jacob (1946)" in gated

    def test_never_offers_an_empty_list(self):
        class Validation:
            suitable_methods = ["Something else entirely"]

        offered, _ = analysis.method_options(Validation())
        assert offered == ["Theis (1935)"]

    def test_help_comes_from_the_registry(self):
        text = analysis.method_help(["Theis (1935)", "Cooper-Jacob (1946)"])
        assert registry.get("theis_1935").ui_help in text

    @pytest.mark.parametrize("suggested,expected", [
        ("Cooper-Jacob (1946)", 1),
        ("cooper jacob", 1),
        ("theis", 0),
        ("", 0),
        ("Boulton (1963)", 0),        # not offered here, so the default stands
    ])
    def test_the_suggested_method_is_preselected(self, suggested, expected):
        options = ["Theis (1935)", "Cooper-Jacob (1946)"]
        assert analysis.default_method_index(options, suggested) == expected


class TestAFinishedResult:

    def test_storativity_is_reported_when_the_method_estimates_it(self, theis_run):
        result, _ = theis_run
        assert analysis.estimates_S(result) is True

    def test_recovery_reports_no_storativity(self):
        df = pd.read_csv(RECOVERY_SAMPLE)
        result = analysis.run_analysis(df, "theis_recovery_1935", _recovery_setup())
        assert analysis.estimates_S(result) is False

    def test_plot_context_carries_the_run_geometry(self, theis_run):
        _, record = theis_run
        ctx = analysis.plot_context(record)

        assert ctx.Q == pytest.approx(500 / 86400) and ctx.r == 50.0

    def test_the_diagnostic_plot_dispatches_by_method(self, theis_run):
        result, record = theis_run
        figure = analysis.diagnostic_plot(result, record)
        assert len(figure.data) >= 2

    def test_manual_matching_at_the_fit_reproduces_its_rmse(self, theis_run):
        result, record = theis_run
        _, rmse_mm = analysis.manual_match(result, record, result.T, result.S)

        assert rmse_mm == pytest.approx(result.rmse * 1000, rel=1e-3)

    def test_format_ci(self, theis_run):
        result, _ = theis_run
        assert analysis.format_ci(None) == "not available"
        assert "m²/day" in analysis.format_ci(result.T_ci)


class TestSavingAndReporting:

    def _test_with_readings(self, df):
        project = new_project()
        test = project.tests[0]
        data_intake.store_readings(test, df, source_name="sample.csv")
        return project, test

    def test_saving_records_the_method_and_the_setup(self, theis_df, theis_run):
        result, record = theis_run
        _, test = self._test_with_readings(theis_df)

        saved = analysis.save_analysis(test, result, record)

        assert saved.method_id == "theis_1935"
        assert saved.setup.Q_entered == 500.0
        assert test.analyses == [saved]

    def test_the_pdf_is_built_and_named_for_the_method(self, theis_df, theis_run):
        result, record = theis_run
        pdf, filename = analysis.report_pdf(
            result, record, theis_df,
            project_info=ProjectInfo(title="Campus well field"),
            test_label="PW-1")

        assert pdf.startswith(b"%PDF-")
        assert filename == "theis_report.pdf"

    def test_a_pumped_well_report_uses_the_screen_radius(self):
        df = pd.read_csv(PC_SAMPLE)
        setup = _setup(Q_m3s=300 / 86400, r_w_m=0.30, r_c_m=0.20)
        result = analysis.run_analysis(df, "papadopulos_cooper_1967", setup)
        record = analysis.run_record(setup, "papadopulos_cooper_1967", result)

        pdf, filename = analysis.report_pdf(result, record, df)

        assert pdf.startswith(b"%PDF-")
        assert filename == "papadopulos-cooper_report.pdf"


class TestDataIntake:

    def test_prepares_an_uploaded_file(self, theis_df):
        prepared = data_intake.prepare_upload(theis_df)

        assert prepared.is_usable
        assert prepared.inspection.time_col == "time_s"
        assert len(prepared.df) == len(theis_df)
        assert "50 data points" in prepared.summary()

    def test_reads_the_flow_regime_off_the_derivative(self, theis_df):
        prepared = data_intake.prepare(theis_df)

        assert prepared.diagnosis is not None and prepared.diagnosis_error == ""
        assert "Theis (1935)" in prepared.consistent_methods

    def test_a_dataset_too_short_to_diagnose_is_not_a_failure(self, theis_df):
        prepared = data_intake.prepare(theis_df.head(10))

        assert prepared.df is not None
        assert prepared.diagnosis is None or prepared.diagnosis_error == ""

    def test_recovery_data_is_not_flagged_for_falling_drawdown(self):
        df = pd.read_csv(RECOVERY_SAMPLE)

        as_pumping = data_intake.prepare(df)
        as_recovery = data_intake.prepare(df, test_type=registry.RECOVERY)

        assert any("recovery" in note.lower() for note in as_pumping.validation.info)
        assert not any("recovery" in note.lower()
                       for note in as_recovery.validation.info)

    def test_the_suitable_methods_follow_the_test_type(self):
        df = pd.read_csv(RECOVERY_SAMPLE)
        prepared = data_intake.prepare(df, test_type=registry.RECOVERY)

        assert prepared.validation.suitable_methods == ["Theis Recovery (1935)"]

    def test_storing_readings_is_guarded_by_their_fingerprint(self, theis_df):
        test = new_project().tests[0]

        assert data_intake.store_readings(test, theis_df, "a.csv") is True
        assert data_intake.store_readings(test, theis_df, "a.csv") is False
        assert data_intake.store_readings(test, theis_df.iloc[::2], "b.csv") is True
        assert len(test.readings) == len(theis_df.iloc[::2])
