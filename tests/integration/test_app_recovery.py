"""The recovery test type in the real Streamlit app, and the method list it drives.

The method dropdown used to be filtered by a validator that only knew the first
three methods, so Hantush-Jacob, Boulton and Warren-Root disappeared from it the
moment a file was uploaded. It is now driven by the registry for the chosen test
type; these tests pin both halves of that.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.project.model import Project, ProjectInfo, Readings, TestSetup
from src.project.workspace import inputs_from_setup

pytestmark = pytest.mark.slow

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(Path(__file__).resolve().parents[2] / "app.py")
TIMEOUT = 240


def _open(csv, setup):
    project = Project(info=ProjectInfo(title="Recovery site"))
    test = project.add_test("PW-1")
    test.set_readings(Readings.from_frame(pd.read_csv(csv), source_name=Path(csv).name))
    test.setup = setup

    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.session_state.project = project
    at.session_state.active_test_id = test.id
    for key, value in inputs_from_setup(setup).items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    return at


def _method_box(at):
    return next(s for s in at.selectbox if s.label == "Analysis method")


class TestMethodList:

    def test_constant_rate_offers_every_constant_rate_method_after_upload(self):
        at = _open("tests/theis_cooper_jacob_sample.csv",
                   TestSetup(Q_m3s=500 / 86400, Q_entered=500.0, r_m=50.0))
        options = _method_box(at).options

        for label in ("Theis (1935)", "Hantush-Jacob (1955)", "Boulton (1963)",
                      "Warren-Root (1963)"):
            assert label in options
        assert "Theis Recovery (1935)" not in options

    def test_recovery_offers_only_recovery_methods_and_asks_for_t_p(self):
        at = _open("tests/theis_recovery_sample.csv",
                   TestSetup(Q_m3s=500 / 86400, Q_entered=500.0, r_m=50.0,
                             test_type="recovery", t_p_s=86400.0))

        assert _method_box(at).options == ["Theis Recovery (1935)"]
        assert any(n.label.startswith("Pumping duration") for n in at.number_input)


class TestRecoveryRun:

    def test_runs_end_to_end_and_reports_no_storativity(self):
        at = _open("tests/theis_recovery_sample.csv",
                   TestSetup(Q_m3s=500 / 86400, Q_entered=500.0, r_m=50.0,
                             test_type="recovery", t_p_s=86400.0))

        next(b for b in at.button if "Run Analysis" in b.label).click().run()
        assert not at.exception, [str(e.value)[:300] for e in at.exception]

        result = at.session_state.calc_result
        assert result.success, result.error_message
        assert result.T == pytest.approx(2.5e-3, rel=0.03)
        assert any(m.label == "S" and m.value == "not determined" for m in at.metric)
        assert at.session_state.calc_inputs["t_p_s"] == pytest.approx(86400.0)

    def test_the_recovery_setup_is_saved_with_the_analysis(self):
        at = _open("tests/theis_recovery_sample.csv",
                   TestSetup(Q_m3s=500 / 86400, Q_entered=500.0, r_m=50.0,
                             test_type="recovery", t_p_s=86400.0))
        next(b for b in at.button if "Run Analysis" in b.label).click().run()
        next(b for b in at.button if b.label.startswith("📌 Save")).click().run()

        saved = at.session_state.project.tests[0].analyses[0]
        assert saved.setup.test_type == "recovery"
        assert saved.setup.t_p_s == pytest.approx(86400.0)
