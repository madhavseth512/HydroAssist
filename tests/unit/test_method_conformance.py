"""Conformance: every registered method must work end to end, not just fit.

Hantush-Jacob, Boulton and Warren-Root each passed their own tests — correct
parameters, correct analytical limits — and still shipped broken. None of them
registered plot adapters, so after a successful fit the workspace's diagnostic
plot, its manual-matching plot and the PDF report all crashed with
"'NoneType' object is not callable". Each method's tests checked the numbers; no
test checked that the product could display them.

These tests are parametrised over the registry itself, so every method — including
ones added later — is automatically held to the same bar: fit, plot, manual-match,
and produce a PDF that names every estimated parameter.
"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.calculator import registry
from src.calculator.base import CalculationInput
from src.calculator.report import generate_pdf

Q_OBS, R_OBS = 500 / 86400, 50.0
Q_PW, R_W, R_C = 300 / 86400, 0.30, 0.20

SPECS = registry.all_specs()


def _sliders(spec, result) -> tuple:
    """What the manual-matching sliders start on: T and S for a pumping test, and
    a method's own `manual_parameters` otherwise."""
    if spec.manual_parameters == ("T", "S"):
        return result.T, result.S
    return tuple(result.value(name) for name in spec.manual_parameters)


def _fit(spec):
    """Fit the method to the bundled sample appropriate to its test type and
    data source."""
    if registry.TRACER in spec.test_types:
        from src.calculator.geometry import TracerGeometry
        from src.services.data_intake import prepare_upload

        path = ("tests/tracer_convergent_sample.csv" if spec.uses_pumping_rate
                else "tests/tracer_natural_sample.csv")
        df = prepare_upload(pd.read_csv(path)).df
        Q = Q_OBS if spec.uses_pumping_rate else None
        inputs = CalculationInput(df=df, Q=Q, r=R_OBS, test_type=registry.TRACER,
                                  tracer=TracerGeometry())
        ctx = registry.PlotContext(Q=Q or 0.0, r=R_OBS)
    elif registry.RECOVERY in spec.test_types:
        from src.calculator.geometry import PumpingSchedule, RateStep

        # Same aquifer and rate as the constant-rate sample, 24 h of pumping.
        df = pd.read_csv("tests/theis_recovery_sample.csv")
        inputs = CalculationInput(
            df=df, Q=Q_OBS, r=R_OBS, test_type=registry.RECOVERY,
            schedule=PumpingSchedule(steps=(RateStep(0.0, Q_OBS),
                                            RateStep(86400.0, 0.0))),
        )
        ctx = registry.PlotContext(Q=Q_OBS, r=R_OBS)
    elif spec.data_source == registry.PUMPED_WELL:
        df = pd.read_csv("tests/papadopulos_cooper_sample.csv")
        inputs = CalculationInput(df=df, Q=Q_PW, r=R_W, r_w=R_W, r_c=R_C)
        ctx = registry.PlotContext(Q=Q_PW, r=R_W, r_w=R_W, r_c=R_C)
    else:
        df = pd.read_csv("tests/theis_cooper_jacob_sample.csv")
        inputs = CalculationInput(df=df, Q=Q_OBS, r=R_OBS)
        ctx = registry.PlotContext(Q=Q_OBS, r=R_OBS, n_excluded=3)
    return spec.build().calculate(inputs), df, ctx


@pytest.fixture(scope="module", params=SPECS, ids=lambda s: s.id)
def fitted(request):
    spec = request.param
    result, df, ctx = _fit(spec)
    assert result.success, f"{spec.label}: {result.error_message}"
    return spec, result, df, ctx


class TestEveryMethodIsDisplayable:

    def test_every_spec_has_all_three_plot_adapters(self):
        """The registry default makes a missing adapter impossible; this pins it."""
        for spec in SPECS:
            assert callable(spec.diagnostic_plot), spec.id
            assert callable(spec.interactive_plot), spec.id
            assert callable(spec.report_plot), spec.id

    def test_diagnostic_plot_renders(self, fitted):
        spec, result, _, ctx = fitted
        fig = spec.diagnostic_plot(result, ctx)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2, "observed data and a fitted curve at least"

    def test_manual_matching_renders(self, fitted):
        spec, result, _, ctx = fitted
        fig, rmse_mm = spec.interactive_plot(result, ctx, *_sliders(spec, result))

        assert isinstance(fig, go.Figure)
        assert np.isfinite(rmse_mm) and rmse_mm >= 0

    def test_manual_matching_at_the_fit_reproduces_the_fit_rmse(self, fitted):
        """With the sliders on the fitted T and S — and every other parameter
        held at its fitted value — the manual curve IS the fit, so its RMSE must
        equal the calculator's. Cooper-Jacob excepted: its RMSE is computed on
        the late-time points it fitted, not the whole record."""
        spec, result, _, ctx = fitted
        if spec.id == "cooper_jacob_1946":
            pytest.skip("RMSE basis differs by design")
        _, rmse_mm = spec.interactive_plot(result, ctx, *_sliders(spec, result))

        assert rmse_mm == pytest.approx(result.rmse * 1000, rel=1e-3)

    def test_moving_the_sliders_changes_the_fit(self, fitted):
        spec, result, _, ctx = fitted
        first, second = _sliders(spec, result)
        _, at_fit = spec.interactive_plot(result, ctx, first, second)
        _, off_fit = spec.interactive_plot(result, ctx, first * 4, second)

        assert off_fit > at_fit

    def test_pdf_report_generates(self, fitted):
        spec, result, df, ctx = fitted
        at_well = spec.data_source == registry.PUMPED_WELL
        pdf = generate_pdf(result, ctx.Q, R_W if at_well else R_OBS, spec.label, df,
                           n_excluded=ctx.n_excluded, r_c=R_C if at_well else None)

        assert pdf.startswith(b"%PDF-")

    def test_pdf_lists_every_estimated_parameter(self, fitted):
        """The results table used to hardcode T and S, silently dropping S_y and
        alpha from Boulton reports, B from Hantush-Jacob, omega and lambda from
        Warren-Root."""
        from pypdf import PdfReader
        from src.calculator.report import _PARAMETER_NAMES

        spec, result, df, ctx = fitted
        at_well = spec.data_source == registry.PUMPED_WELL
        pdf = generate_pdf(result, ctx.Q, R_W if at_well else R_OBS, spec.label, df,
                           n_excluded=ctx.n_excluded, r_c=R_C if at_well else None)
        text = " ".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)

        for p in result.parameters:
            assert _PARAMETER_NAMES.get(p.name, p.name) in text, (
                f"{spec.label}: parameter {p.name!r} missing from the PDF")

    def test_pdf_header_names_the_right_aquifer_type(self, fitted):
        """The subtitle used to say "Confined" for every method."""
        from pypdf import PdfReader

        spec, result, df, ctx = fitted
        at_well = spec.data_source == registry.PUMPED_WELL
        pdf = generate_pdf(result, ctx.Q, R_W if at_well else R_OBS, spec.label, df,
                           n_excluded=ctx.n_excluded, r_c=R_C if at_well else None)
        first_page = PdfReader(io.BytesIO(pdf)).pages[0].extract_text()

        for aquifer in spec.aquifer_types:
            assert aquifer.title() in first_page, spec.id

    def test_pdf_footer_cites_the_method_used(self, fitted):
        """The footer used to credit Theis and Cooper-Jacob on every report."""
        from pypdf import PdfReader

        spec, result, df, ctx = fitted
        at_well = spec.data_source == registry.PUMPED_WELL
        pdf = generate_pdf(result, ctx.Q, R_W if at_well else R_OBS, spec.label, df,
                           n_excluded=ctx.n_excluded, r_c=R_C if at_well else None)
        text = " ".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)

        assert str(spec.year) in text and spec.authors[0].split(",")[0] in text


class TestProjectMetadataInTheReport:

    def test_project_details_head_the_report(self):
        from pypdf import PdfReader
        from src.project.model import ProjectInfo

        spec = registry.get("theis_1935")
        result, df, ctx = _fit(spec)
        info = ProjectInfo(title="Campus well field", client="IIT Kharagpur",
                           project_number="BTP-26-04", location="Kharagpur",
                           test_date="2026-09-01", analyst="M. Seth")
        pdf = generate_pdf(result, ctx.Q, R_OBS, spec.label, df,
                           project_info=info, test_label="PW-1 constant rate")
        first_page = PdfReader(io.BytesIO(pdf)).pages[0].extract_text()

        for value in ("Campus well field", "IIT Kharagpur", "BTP-26-04",
                      "Kharagpur", "2026-09-01", "PW-1 constant rate"):
            assert value in first_page, value

    def test_no_project_means_no_project_block(self):
        from pypdf import PdfReader

        spec = registry.get("theis_1935")
        result, df, ctx = _fit(spec)
        pdf = generate_pdf(result, ctx.Q, R_OBS, spec.label, df)
        first_page = PdfReader(io.BytesIO(pdf)).pages[0].extract_text()

        assert "Client" not in first_page
