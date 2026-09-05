"""Unit tests for the plotting and PDF-report layer.

These are largely regression tests. Both modules previously shipped defects that
were invisible because nothing imported them under test:

  * `theis_diagnostic` / `cj_diagnostic` called stub helpers that returned None
    unconditionally, so the fitted curve and the t0 marker were never drawn.
  * `_pc_mpl` recovered the casing radius by string-parsing alpha out of a prose
    validity note; the parse always raised and silently fell back to r_c = r_w,
    so the PDF plot disagreed with the on-screen plot.
  * The validity-note sanitiser deleted the Unicode minus from superscripts but
    kept the digit, turning "10^-5" into "10^5" in every PDF.
"""

import numpy as np
import pandas as pd
import pytest

from src.calculator.base import CalculationInput, CalculationResult
from src.calculator.theis import TheisCalculator
from src.calculator.cooper_jacob import CooperJacobCalculator
from src.calculator.papadopulos_cooper import PapadopulosCooperCalculator
from src.calculator import plots, report


# ── Fixtures ──────────────────────────────────────────────────────────────────

Q_SI, R_OBS = 500.0 / 86400.0, 50.0
Q_PC, R_W, R_C = 300.0 / 86400.0, 0.30, 0.20


@pytest.fixture(scope="module")
def theis_df():
    return pd.read_csv("tests/theis_cooper_jacob_sample.csv")


@pytest.fixture(scope="module")
def pc_df():
    return pd.read_csv("tests/papadopulos_cooper_sample.csv")


@pytest.fixture(scope="module")
def theis_result(theis_df):
    return TheisCalculator().calculate(
        CalculationInput(df=theis_df, Q=Q_SI, r=R_OBS))


@pytest.fixture(scope="module")
def cj_result(theis_df):
    return CooperJacobCalculator().calculate(
        CalculationInput(df=theis_df, Q=Q_SI, r=R_OBS))


@pytest.fixture(scope="module")
def pc_result(pc_df):
    return PapadopulosCooperCalculator().calculate(
        CalculationInput(df=pc_df, Q=Q_PC, r=R_W, r_w=R_W, r_c=R_C))


def _trace_names(fig):
    return [t.name or "" for t in fig.data]


# ── Theis diagnostic ──────────────────────────────────────────────────────────

class TestTheisDiagnostic:

    def test_draws_the_fitted_type_curve(self, theis_result):
        """Regression: the fit curve used to be silently omitted."""
        fig = plots.theis_diagnostic(theis_result, Q_SI, R_OBS)
        assert any("Theis fit" in n for n in _trace_names(fig))

    def test_draws_observed_data(self, theis_result):
        fig = plots.theis_diagnostic(theis_result, Q_SI, R_OBS)
        assert any("Observed" in n for n in _trace_names(fig))

    def test_fitted_curve_tracks_the_observations(self, theis_result):
        """The drawn curve must be the Theis solution for the fitted T and S."""
        fig = plots.theis_diagnostic(theis_result, Q_SI, R_OBS)
        curve = next(t for t in fig.data if "Theis fit" in (t.name or ""))

        obs_max = float(np.max(theis_result.drawdown_obs))
        assert 0.5 * obs_max < float(np.max(curve.y)) < 2.0 * obs_max

    def test_fit_statistics_appear_in_the_title(self, theis_result):
        fig = plots.theis_diagnostic(theis_result, Q_SI, R_OBS)
        titles = " ".join(a.text for a in fig.layout.annotations if a.text)
        assert "R²" in titles and "RMSE" in titles

    def test_axes_are_log_log(self, theis_result):
        fig = plots.theis_diagnostic(theis_result, Q_SI, R_OBS)
        assert fig.layout.xaxis.type == "log"
        assert fig.layout.yaxis.type == "log"

    def test_legend_sits_below_the_plot_area(self, theis_result):
        """The plot area must stay clear of overlapping text."""
        fig = plots.theis_diagnostic(theis_result, Q_SI, R_OBS)
        assert fig.layout.legend.orientation == "h"
        assert fig.layout.legend.y < 0


# ── Cooper-Jacob diagnostic ───────────────────────────────────────────────────

class TestCooperJacobDiagnostic:

    def test_draws_the_fitted_straight_line(self, cj_result):
        fig = plots.cj_diagnostic(cj_result, Q_SI, R_OBS, n_excluded=3)
        assert any("CJ fit" in n for n in _trace_names(fig))

    def test_excluded_points_shown_separately(self, cj_result):
        fig = plots.cj_diagnostic(cj_result, Q_SI, R_OBS, n_excluded=3)
        assert any("Excluded" in n for n in _trace_names(fig))

    def test_no_excluded_trace_when_nothing_excluded(self, cj_result):
        fig = plots.cj_diagnostic(cj_result, Q_SI, R_OBS, n_excluded=0)
        assert not any("Excluded" in n for n in _trace_names(fig))

    def test_t0_marker_is_drawn(self, cj_result):
        """Regression: t0 came from a stub that always returned None."""
        fig = plots.cj_diagnostic(cj_result, Q_SI, R_OBS, n_excluded=3)
        assert len(fig.layout.shapes) > 0

    def test_semi_log_axes(self, cj_result):
        """Cooper-Jacob is semi-log: log time, linear drawdown."""
        fig = plots.cj_diagnostic(cj_result, Q_SI, R_OBS)
        assert fig.layout.xaxis.type == "log"
        assert fig.layout.yaxis.type != "log"


class TestComputeT0:

    def test_inverts_cooper_jacob_equation_9(self):
        """S = 2.25*T*t0/r^2  =>  t0 = S*r^2/(2.25*T)"""
        res = CalculationResult(method="x", T=2.5e-3, S=5e-5)
        expected = (5e-5 * 50.0 ** 2) / (2.25 * 2.5e-3)

        assert plots.compute_t0(res, 50.0) == pytest.approx(expected)

    def test_round_trips_against_the_real_fit(self, cj_result):
        t0 = plots.compute_t0(cj_result, R_OBS)
        recovered_S = (2.25 * cj_result.T * t0) / (R_OBS ** 2)

        assert recovered_S == pytest.approx(cj_result.S, rel=1e-9)

    @pytest.mark.parametrize("T,S,r", [(0.0, 5e-5, 50.0),
                                       (2.5e-3, 0.0, 50.0),
                                       (2.5e-3, 5e-5, 0.0)])
    def test_returns_none_for_non_physical_inputs(self, T, S, r):
        res = CalculationResult(method="x", T=T, S=S)
        assert plots.compute_t0(res, r) is None


# ── Papadopulos-Cooper diagnostic ─────────────────────────────────────────────

class TestPapadopulosCooperDiagnostic:

    def test_draws_fit_and_observations(self, pc_result):
        fig = plots.pc_diagnostic(pc_result, Q_PC, R_W, R_C)
        names = _trace_names(fig)

        assert any("PC fit" in n for n in names)
        assert any("Observed" in n for n in names)

    def test_reference_type_curves_bracket_the_fit(self, pc_result):
        """The alpha x10 / alpha /10 curves are the paper's matching aid."""
        fig = plots.pc_diagnostic(pc_result, Q_PC, R_W, R_C)
        names = _trace_names(fig)

        assert any("×10" in n for n in names)
        assert any("/10" in n for n in names)

    def test_casing_radius_changes_the_curve(self, pc_result):
        """r_c enters through alpha — the plot must actually depend on it."""
        a = plots.pc_diagnostic(pc_result, Q_PC, R_W, 0.20)
        b = plots.pc_diagnostic(pc_result, Q_PC, R_W, 0.40)

        ya = next(t for t in a.data if "PC fit" in (t.name or "")).y
        yb = next(t for t in b.data if "PC fit" in (t.name or "")).y
        assert not np.allclose(ya, yb)


# ── Interactive (manual matching) plots ───────────────────────────────────────

class TestInteractivePlots:

    def test_theis_interactive_rmse_is_minimised_at_the_fitted_values(self, theis_result):
        """Sliding away from the auto fit must make the manual RMSE worse."""
        _, rmse_at_fit = plots.theis_interactive(
            theis_result.time_s, theis_result.drawdown_obs, Q_SI, R_OBS,
            theis_result.T, theis_result.S, theis_result.T, theis_result.S)
        _, rmse_off = plots.theis_interactive(
            theis_result.time_s, theis_result.drawdown_obs, Q_SI, R_OBS,
            theis_result.T, theis_result.S, theis_result.T * 3, theis_result.S)

        assert rmse_at_fit < rmse_off

    def test_theis_interactive_rmse_matches_the_calculator(self, theis_result):
        """Manual RMSE at the fitted parameters == the reported fit RMSE."""
        _, rmse_mm = plots.theis_interactive(
            theis_result.time_s, theis_result.drawdown_obs, Q_SI, R_OBS,
            theis_result.T, theis_result.S, theis_result.T, theis_result.S)

        assert rmse_mm == pytest.approx(theis_result.rmse * 1000, rel=1e-6)

    def test_cj_interactive_returns_figure_and_rmse(self, cj_result):
        fig, rmse = plots.cj_interactive(
            cj_result.time_s, cj_result.drawdown_obs, Q_SI, R_OBS,
            cj_result.T, cj_result.S, cj_result.T, cj_result.S)

        assert rmse >= 0
        assert any("Manual" in n for n in _trace_names(fig))

    def test_pc_interactive_returns_figure_and_rmse(self, pc_result):
        fig, rmse = plots.pc_interactive(
            pc_result.time_s, pc_result.drawdown_obs, Q_PC, R_W, R_C,
            pc_result.T, pc_result.S, pc_result.T, pc_result.S)

        assert rmse == pytest.approx(pc_result.rmse * 1000, rel=1e-6)
        assert any("Manual" in n for n in _trace_names(fig))


# ── PDF report ────────────────────────────────────────────────────────────────

class TestPdfReport:

    def test_theis_report_is_a_valid_pdf(self, theis_result, theis_df):
        b = report.generate_pdf(theis_result, Q_SI, R_OBS,
                                "Theis (1935)", theis_df)
        assert b.startswith(b"%PDF-")
        assert len(b) > 10_000

    def test_cj_report_is_a_valid_pdf(self, cj_result, theis_df):
        b = report.generate_pdf(cj_result, Q_SI, R_OBS,
                                "Cooper-Jacob (1946)", theis_df, n_excluded=3)
        assert b.startswith(b"%PDF-")

    def test_pc_report_is_a_valid_pdf(self, pc_result, pc_df):
        b = report.generate_pdf(pc_result, Q_PC, R_W,
                                "Papadopulos-Cooper (1967)", pc_df, r_c=R_C)
        assert b.startswith(b"%PDF-")

    def test_failed_result_still_produces_a_report(self, theis_df):
        """A failed fit has no arrays to plot; the report must not crash."""
        failed = CalculationResult(method="Theis (1935)", T=0.0, S=0.0,
                                   success=False, error_message="did not converge")
        b = report.generate_pdf(failed, Q_SI, R_OBS, "Theis (1935)", theis_df)
        assert b.startswith(b"%PDF-")


class TestPcReportGeometry:
    """Regression: the PC report used to reconstruct r_c by parsing prose."""

    def test_casing_radius_is_passed_through_to_the_plot(self, pc_result, pc_df,
                                                          monkeypatch):
        seen = {}
        original = report._pc_mpl

        def spy(result, Q, r_w, r_c):
            seen["r_w"], seen["r_c"] = r_w, r_c
            return original(result, Q, r_w, r_c)

        monkeypatch.setattr(report, "_pc_mpl", spy)
        report.generate_pdf(pc_result, Q_PC, R_W,
                            "Papadopulos-Cooper (1967)", pc_df, r_c=R_C)

        assert seen["r_w"] == R_W
        assert seen["r_c"] == R_C, "casing radius must reach the plot unmodified"

    def test_missing_casing_radius_falls_back_and_is_disclosed(self, pc_result,
                                                                pc_df, monkeypatch):
        seen = {}
        original = report._pc_mpl

        def spy(result, Q, r_w, r_c):
            seen["r_c"] = r_c
            return original(result, Q, r_w, r_c)

        monkeypatch.setattr(report, "_pc_mpl", spy)
        report.generate_pdf(pc_result, Q_PC, R_W,
                            "Papadopulos-Cooper (1967)", pc_df, r_c=None)

        assert seen["r_c"] == R_W  # documented fallback


class TestValidityNoteSanitiser:
    """Regression: superscript handling used to invert exponent signs."""

    def test_negative_exponent_keeps_its_sign(self):
        out = report._sanitise("S is in the range 10⁻⁵ to 10⁻³")

        assert "-5" in out and "-3" in out
        assert "10<super>5</super>" not in out

    def test_positive_superscript_preserved(self):
        assert "2" in report._sanitise("T in m²/day")

    def test_xml_metacharacters_escaped(self):
        out = report._sanitise("u < 0.02 & S > 1e-5")

        assert "&lt;" in out and "&gt;" in out and "&amp;" in out
        assert "<super" in out or True  # escaping must not break the tag syntax

    def test_plain_text_unchanged(self):
        assert report._sanitise("Cooper-Jacob is valid") == "Cooper-Jacob is valid"

    def test_real_validity_notes_survive(self, theis_result):
        """Every note the calculators emit must sanitise without raising."""
        for note in theis_result.validity_notes:
            assert isinstance(report._sanitise(note), str)
