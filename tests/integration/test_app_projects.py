"""Projects in the real Streamlit app.

Rendered through AppTest with real config, store and calculators. File uploads
cannot be driven through AppTest, so a reopened project is simulated the way the
app itself lands after opening one: the project in session state, its first test
active, and the Step 2 widget keys prefilled from that test's setup. Opening and
parsing the file is covered by the unit tests of Project.from_json.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.calculator.base import CalculationInput
from src.calculator.methods.hantush_jacob import (
    HantushJacobCalculator, hantush_jacob_drawdown,
)
from src.calculator.theis import TheisCalculator
from src.project.model import Project, ProjectInfo, Readings, TestSetup
from src.project.workspace import inputs_from_setup

pytestmark = pytest.mark.slow

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(Path(__file__).resolve().parents[2] / "app.py")
TIMEOUT = 240
Q, R = 500 / 86400, 50.0


def _theis_df():
    return pd.read_csv("tests/theis_cooper_jacob_sample.csv")


def _leaky_df():
    t = np.logspace(1, 6, 70)
    s = hantush_jacob_drawdown(t, 2.5e-3, 5e-5, 300.0, Q, R)
    s = s + np.random.default_rng(2).normal(0, 0.002, t.size)
    return pd.DataFrame({"time_s": t, "drawdown_m": s})


def _project(df, q_day=300.0, r=40.0, name="PW-1"):
    project = Project(info=ProjectInfo(title="Reopened site"))
    test = project.add_test(name)
    test.set_readings(Readings.from_frame(df, source_name="pw1.csv"))
    test.setup = TestSetup(Q_m3s=q_day / 86400, Q_entered=q_day,
                           Q_unit="m³/day", r_m=r)
    return project, test


def _open(project, test):
    """Land the app where it lands after opening a project file."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.session_state.project = project
    at.session_state.active_test_id = test.id
    at.session_state.formatted_df = None
    for key, value in inputs_from_setup(test.setup).items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    return at


def _number_input(at, label):
    return next(n for n in at.number_input if n.label == label)


class TestProjectSidebar:

    def test_project_section_renders(self):
        at = AppTest.from_file(APP, default_timeout=TIMEOUT)
        at.run()

        assert not at.exception
        assert "📁 Project" in [h.value for h in at.sidebar.subheader]

    def test_a_fresh_session_starts_with_one_empty_test(self):
        at = AppTest.from_file(APP, default_timeout=TIMEOUT)
        at.run()

        project = at.session_state.project
        assert len(project.tests) == 1
        assert project.tests[0].readings is None


class TestReopeningAProject:

    def test_saved_readings_are_used_without_a_new_upload(self):
        at = _open(*_project(_theis_df()))

        assert at.session_state.formatted_df is not None
        assert len(at.session_state.formatted_df) == len(_theis_df())
        assert any("readings saved in this project from pw1.csv" in i.value
                   for i in at.info)

    def test_run_is_enabled_by_reopened_readings(self):
        at = _open(*_project(_theis_df()))
        run = next(b for b in at.button if "Run Analysis" in b.label)

        assert run.disabled is False

    def test_the_form_comes_back_as_it_was_saved(self):
        at = _open(*_project(_theis_df(), q_day=300.0, r=40.0))

        assert _number_input(at, "Pumping rate Q").value == pytest.approx(300.0)
        assert _number_input(at, "Observation well distance r (m)").value == \
            pytest.approx(40.0)

    def test_the_diagnostic_runs_on_reopened_readings(self):
        at = _open(*_project(_theis_df()))
        assert at.session_state.diagnosis is not None


class TestSavingAnalyses:

    def test_a_result_can_be_saved_to_the_active_test(self):
        df = _theis_df()
        project, test = _project(df, q_day=500.0, r=50.0)
        at = _open(project, test)

        result = TheisCalculator().calculate(CalculationInput(df=df, Q=Q, r=R))
        at.session_state.calc_result = result
        at.session_state.calc_inputs = {
            "Q": Q, "r": R, "r_w": 0.10, "r_c": 0.10, "method": "Theis (1935)",
            "n_excluded": 0, "Q_entered": 500.0, "Q_unit": "m³/day",
        }
        at.run()

        save = next(b for b in at.button if b.label.startswith("📌 Save this analysis"))
        save.click().run()

        saved = at.session_state.project.get_test(test.id).analyses
        assert len(saved) == 1
        assert saved[0].method_id == "theis_1935"
        assert saved[0].setup.Q_entered == 500.0, "the run's setup, not the form's"

    def test_a_saved_result_cannot_be_saved_twice(self):
        df = _theis_df()
        project, test = _project(df, q_day=500.0, r=50.0)
        at = _open(project, test)
        at.session_state.calc_result = TheisCalculator().calculate(
            CalculationInput(df=df, Q=Q, r=R))
        at.session_state.calc_inputs = {"Q": Q, "r": R, "method": "Theis (1935)",
                                        "n_excluded": 0}
        at.run()
        next(b for b in at.button if b.label.startswith("📌 Save")).click().run()

        assert not any(b.label.startswith("📌 Save") for b in at.button)
        assert any("Saved to" in s.value for s in at.success)


class TestComparison:

    def test_saved_analyses_are_ranked_by_aicc(self):
        df = _leaky_df()
        project, test = _project(df, q_day=500.0, r=50.0)
        for calc, label in ((TheisCalculator, "Theis (1935)"),
                            (HantushJacobCalculator, "Hantush-Jacob (1955)")):
            test.add_analysis(calc().calculate(CalculationInput(df=df, Q=Q, r=R)),
                              label)
        at = _open(project, test)

        tables = [f.value for f in at.dataframe if "ΔAICc" in f.value.columns]
        assert tables, "the comparison table should render"
        assert list(tables[0]["Method"]) == ["Hantush-Jacob (1955)", "Theis (1935)"]

    def test_no_table_without_saved_analyses(self):
        at = _open(*_project(_theis_df()))
        assert not any("ΔAICc" in f.value.columns for f in at.dataframe)


class TestSwitchingTests:

    def test_selecting_another_test_loads_its_data_and_setup(self):
        project, first = _project(_theis_df(), q_day=300.0, r=40.0, name="PW-1")
        second = project.add_test("PW-2")
        second.set_readings(Readings.from_frame(_leaky_df(), source_name="pw2.csv"))
        second.setup = TestSetup(Q_m3s=750 / 86400, Q_entered=750.0,
                                 Q_unit="m³/day", r_m=80.0)
        at = _open(project, first)

        box = next(s for s in at.selectbox if s.label == "Active test")
        box.set_value(second.id).run()

        assert at.session_state.active_test_id == second.id
        assert _number_input(at, "Pumping rate Q").value == pytest.approx(750.0)
        assert len(at.session_state.formatted_df) == len(_leaky_df())

    def test_adding_a_test_switches_to_it(self):
        at = AppTest.from_file(APP, default_timeout=TIMEOUT)
        at.run()
        next(b for b in at.button if "Add test" in b.label).click().run()

        project = at.session_state.project
        assert len(project.tests) == 2
        assert at.session_state.active_test_id == project.tests[1].id


class TestStartingANewProject:

    def test_discarding_replaces_the_project_and_clears_the_workspace(self):
        project, test = _project(_theis_df(), q_day=300.0, r=40.0)
        at = _open(project, test)

        next(b for b in at.button if b.label == "Discard and start new").click().run()

        fresh = at.session_state.project
        assert fresh is not project
        assert len(fresh.tests) == 1 and fresh.tests[0].readings is None
        assert at.session_state.formatted_df is None
        assert _number_input(at, "Pumping rate Q").value == pytest.approx(500.0), \
            "the form returns to defaults"
