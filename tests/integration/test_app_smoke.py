"""Streamlit UI smoke tests via the AppTest harness.

These run app.py for real — real config, real ChromaDB, real embedding model —
and assert on the rendered widget tree. They are the only tests that would catch
a NameError or a bad widget argument in the UI layer, which unit tests on
src/ cannot see.

They are slow (the embedding model loads on first run), so they are marked
`slow`. Run just the fast suite with:  pytest -m "not slow"
"""

from pathlib import Path

import pandas as pd
import pytest

from src.agents.method_selection import THEIS, COOPER_JACOB, PAPADOPULOS_COOPER
from src.calculator.base import CalculationInput
from src.calculator.theis import TheisCalculator

pytestmark = pytest.mark.slow

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

# Absolute, because Streamlit 1.63 changed AppTest.from_file() to resolve a
# relative path against the calling test file rather than the working
# directory — "app.py" silently became tests/integration/app.py and every UI
# test failed before the app booted. This form works under both behaviours.
APP = str(Path(__file__).resolve().parents[2] / "app.py")
TIMEOUT = 240


@pytest.fixture(scope="module")
def app():
    """One booted app instance shared by the read-only assertions."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    return at


def _fresh():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    return at


def _fitted_result():
    df = pd.read_csv("tests/theis_cooper_jacob_sample.csv")
    result = TheisCalculator().calculate(
        CalculationInput(df=df, Q=500 / 86400, r=50.0))
    return result, df


# ── Boot ──────────────────────────────────────────────────────────────────────

class TestAppBoots:

    def test_runs_without_exception(self, app):
        assert not app.exception, [str(e.value) for e in app.exception]

    def test_both_tabs_render(self, app):
        assert len(app.tabs) == 2

    def test_calculator_workflow_steps_are_present(self, app):
        headings = [s.value for s in app.subheader]

        assert any("Step 1" in h for h in headings)
        assert any("Step 2" in h for h in headings)
        assert any("Step 3" in h for h in headings)

    def test_every_constant_rate_method_is_offered_before_any_upload(self, app):
        """The default test type is constant-rate; recovery methods appear only
        when the Recovery test type is chosen."""
        from src.calculator import registry

        method_box = next(s for s in app.selectbox if s.label == "Analysis method")
        assert list(method_box.options) == [
            s.label for s in registry.for_test_type(registry.CONSTANT_RATE)]

    def test_pumping_rate_units_offered(self, app):
        q_box = next(s for s in app.selectbox if s.label == "Q unit")
        assert "m³/day" in q_box.options
        assert "L/s" in q_box.options

    def test_run_button_disabled_without_data(self, app):
        run = [b for b in app.button if "Run Analysis" in b.label]
        assert run, "Run Analysis button should be rendered"
        assert run[0].disabled is True


# ── The suggestion card (chat → calculator) ───────────────────────────────────

class TestReportDownload:
    """The PDF button used to read `spec_r` about thirty lines before that
    variable was assigned, so every click raised NameError and the surrounding
    try/except reported "Failed to generate PDF". No test clicked it. The report
    is now assembled by src/services/analysis.report_pdf."""

    def test_generating_the_report_offers_a_download(self):
        result, df = _fitted_result()
        at = _fresh()
        at.session_state.calc_result = result
        at.session_state.formatted_df = df
        at.session_state.calc_inputs = {
            "Q": 500 / 86400, "r": 50.0, "method": THEIS, "n_excluded": 0,
            "r_w": 0.10, "r_c": 0.10,
        }
        at.run()

        next(b for b in at.button if "Generate PDF Report" in b.label).click().run()

        assert not at.exception, [str(e.value)[:300] for e in at.exception]
        assert not any("Failed to generate PDF" in e.value for e in at.error)
        # AppTest has no typed download_button accessor in this version.
        labels = [getattr(d, "label", "") for d in at.get("download_button")]
        assert any("Download PDF Report" in label for label in labels), labels


class TestSessionsAreIsolated:
    """Two visitors must never share a conversation. The chat object used to be
    st.cache_resource'd, so a method confirmed in one session hid the suggestion
    card in every other session in the process — and on a hosted deployment,
    every visitor would have shared one chat history."""

    def test_each_session_gets_its_own_chat(self):
        first, second = _fresh(), _fresh()
        assert first.session_state.chat is not second.session_state.chat

    def test_confirming_a_method_in_one_session_does_not_leak(self):
        first = _fresh()
        first.session_state.chat.confirm_method(COOPER_JACOB)

        second = _fresh()
        assert not second.session_state.chat.state.get("suggestion_confirmed")


class TestSuggestionCard:

    def test_card_appears_when_the_manager_suggests_a_method(self):
        at = _fresh()
        at.session_state.chat.state["method_suggestion"] = {
            "method": PAPADOPULOS_COOPER,
            "reason": "Large-diameter well; wellbore storage matters.",
            "confidence": 0.85,
            "supported": True,
        }
        at.run()

        text = " ".join(m.value for m in at.markdown)
        assert PAPADOPULOS_COOPER in text
        assert "wellbore storage" in text.lower()
        assert any("Use this method" in b.label for b in at.button)

    def test_unsupported_aquifer_renders_a_warning_not_a_card(self):
        at = _fresh()
        at.session_state.chat.state["method_suggestion"] = {
            "method": "",
            "reason": "No calculator is implemented for unconfined aquifers yet.",
            "confidence": 0.9,
            "supported": False,
        }
        at.run()

        assert any("unconfined" in w.value for w in at.warning)
        assert not any("Use this method" in b.label for b in at.button)

    def test_confirming_the_method_sets_state(self):
        at = _fresh()
        at.session_state.chat.state["method_suggestion"] = {
            "method": COOPER_JACOB, "reason": "Late-time data.",
            "confidence": 0.8, "supported": True,
        }
        at.run()

        next(b for b in at.button if "Use this method" in b.label).click().run()

        assert at.session_state.chat.state["suggestion_confirmed"] is True
        assert at.session_state.chat.state["selected_method"] == COOPER_JACOB

    def test_confirmed_method_preselects_the_dropdown(self):
        """The whole point of the bridge: chat decides, the workspace follows."""
        at = _fresh()
        at.session_state.chat.confirm_method(PAPADOPULOS_COOPER)
        at.run()

        method_box = next(s for s in at.selectbox if s.label == "Analysis method")
        assert method_box.value == PAPADOPULOS_COOPER

    def test_card_hidden_once_confirmed(self):
        at = _fresh()
        at.session_state.chat.state["method_suggestion"] = {
            "method": THEIS, "reason": "Default.",
            "confidence": 0.75, "supported": True,
        }
        at.session_state.chat.confirm_method(THEIS)
        at.run()

        assert not any("Use this method" in b.label for b in at.button)
        assert any(THEIS in s.value for s in at.success)


# ── Results → chat ────────────────────────────────────────────────────────────

class TestResultsPanel:

    def test_recorded_result_is_advertised_in_the_sidebar(self):
        at = _fresh()
        result, df = _fitted_result()
        at.session_state.chat.record_calculation(
            result.to_state_dict(),
            {"Q": 500 / 86400, "r": 50.0, "method": THEIS, "n_points": len(df)})
        at.run()

        assert any("consultant can discuss" in s.value for s in at.success)

    def test_results_panel_renders_the_fitted_parameters(self):
        at = _fresh()
        result, df = _fitted_result()
        at.session_state.calc_result = result
        at.session_state.formatted_df = df
        at.session_state.calc_inputs = {
            "Q": 500 / 86400, "r": 50.0, "method": THEIS,
            "n_excluded": 0, "r_w": 0.10, "r_c": 0.10,
        }
        at.run()

        assert not at.exception, [str(e.value) for e in at.exception]
        values = [m.value for m in at.metric]
        assert f"{result.T_day:.2f}" in values
        assert any("Interpret these results in chat" in b.label for b in at.button)

    def test_diagnostic_plot_is_rendered_for_a_result(self):
        at = _fresh()
        result, df = _fitted_result()
        at.session_state.calc_result = result
        at.session_state.formatted_df = df
        at.session_state.calc_inputs = {
            "Q": 500 / 86400, "r": 50.0, "method": THEIS,
            "n_excluded": 0, "r_w": 0.10, "r_c": 0.10,
        }
        at.run()

        # The Plotly chart is not a typed AppTest element; assert the surrounding
        # workflow rendered and nothing raised while building the figure.
        headings = [s.value for s in at.subheader]
        assert any("Diagnostic Plots" in h for h in headings)
        assert not at.exception
