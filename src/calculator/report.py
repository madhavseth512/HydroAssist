"""PDF report generator for pumping test analysis results.

Uses reportlab for PDF layout and matplotlib for embedded diagnostic plots
(matplotlib is already installed; avoids the kaleido dependency that Plotly
static export requires).

Public API:
    generate_pdf(result, Q_si, r, method, df, n_excluded) -> bytes
"""

import io
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe in server contexts
import matplotlib.pyplot as plt
from scipy.special import expn

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, KeepTogether,
)

from src.calculator.base import CalculationResult

# ── Colours ───────────────────────────────────────────────────────────────────
BLUE      = colors.HexColor("#1E88E5")
DARK_BLUE = colors.HexColor("#1565C0")
LIGHT_BG  = colors.HexColor("#F5F8FF")
BORDER    = colors.HexColor("#CCCCCC")
RED       = colors.HexColor("#D32F2F")
ORANGE    = colors.HexColor("#E65100")
TEXT      = colors.HexColor("#212121")
MUTED     = colors.HexColor("#757575")

# ── Page geometry ─────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm
USABLE_W = PAGE_W - 2 * MARGIN   # ~17 cm


# ── Public entry point ────────────────────────────────────────────────────────

def generate_pdf(
    result: CalculationResult,
    Q_si: float,
    r: float,
    method: str,
    df: pd.DataFrame,
    n_excluded: int = 0,
    r_c: float = None,
    project_info=None,
    test_label: str = "",
) -> bytes:
    """
    Build a PDF report for a completed pumping test analysis.

    Args:
        result:     CalculationResult from any of the three calculators
        Q_si:       Pumping rate in m³/s
        r:          Observation well distance in m. For Papadopulos-Cooper this
                    is the well screen radius r_w (data come from the pumped well).
        method:     Method label ("Theis (1935)", "Cooper-Jacob (1946)",
                    "Papadopulos-Cooper (1967)")
        df:         Formatted DataFrame (time_s, drawdown_m)
        n_excluded: Number of early-time points excluded (Cooper-Jacob only)
        r_c:        Casing radius in m — REQUIRED for Papadopulos-Cooper, ignored
                    otherwise. Without it the embedded type curve cannot be
                    reproduced, so it falls back to r_c = r_w and the plot
                    caption says so.
        project_info: Optional ProjectInfo (client, project number, location,
                    test date, analyst). The report is a client deliverable, so
                    when a project is open its identifying details head page one.
        test_label: Optional name of the test within the project, e.g.
                    "PW-1 constant rate".

    Returns:
        PDF as bytes — pass directly to st.download_button
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title=f"HydroAssist — {method} Report",
        author="HydroAssist (IIT Kharagpur)",
    )

    styles = _build_styles()
    story = []

    story += _header_section(method, styles)
    story += _project_section(project_info, test_label, styles)
    story += (_tracer_config_section(result, Q_si, r, df, method, styles)
              if _tracer_info(result) else
              _config_section(Q_si, r, df, method, styles))
    story += _results_section(result, styles)
    story += _validity_section(result, styles)
    story += _plot_section(result, Q_si, r, method, n_excluded, styles, r_c)
    story += _data_table_section(df, styles, _tracer_info(result))
    story += _footer_section(method, styles)

    doc.build(story)
    return buf.getvalue()


# ── Style definitions ─────────────────────────────────────────────────────────

def _build_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Normal"],
            fontSize=22, textColor=DARK_BLUE, fontName="Helvetica-Bold",
            spaceAfter=4, alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"],
            fontSize=11, textColor=MUTED, fontName="Helvetica",
            spaceAfter=2, alignment=TA_LEFT,
        ),
        "section": ParagraphStyle(
            "section", parent=base["Normal"],
            fontSize=13, textColor=DARK_BLUE, fontName="Helvetica-Bold",
            spaceBefore=12, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"],
            fontSize=9, textColor=TEXT, fontName="Helvetica",
            spaceAfter=3, leading=13,
        ),
        "note": ParagraphStyle(
            "note", parent=base["Normal"],
            fontSize=8.5, textColor=TEXT, fontName="Helvetica",
            spaceAfter=3, leading=12, leftIndent=8,
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["Normal"],
            fontSize=8, textColor=MUTED, fontName="Helvetica-Oblique",
            spaceAfter=4, alignment=TA_CENTER,
        ),
    }


# ── Section builders ──────────────────────────────────────────────────────────

def _aquifer_phrase(method: str) -> str:
    """The aquifer type(s) the method is for, from its registry entry."""
    from src.calculator import registry

    spec = registry.get(method)
    if spec is None or not spec.aquifer_types:
        return ""
    return " / ".join(sorted(spec.aquifer_types)).title() + " "


def _tracer_info(result: CalculationResult) -> dict:
    """What a tracer result carries about its test; empty for any other method."""
    return result.curves.get("tracer") or {}


def _test_kind(method: str) -> str:
    from src.calculator import registry

    spec = registry.get(method)
    if spec is not None and registry.TRACER in spec.test_types:
        return "Tracer Test"
    return "Pumping Test"


def _header_section(method: str, s: dict) -> list:
    now = datetime.now().strftime("%Y-%m-%d  %H:%M")
    return [
        Paragraph("HydroAssist", s["title"]),
        # Was hardcoded "Confined Aquifer..." — wrong once leaky, unconfined and
        # fractured methods existed. Derived from the method's registry entry.
        Paragraph(f"{_aquifer_phrase(method)}Aquifer {_test_kind(method)} Analysis",
                  s["subtitle"]),
        Paragraph(f"Method: <b>{method}</b>  |  Generated: {now}", s["subtitle"]),
        HRFlowable(width=USABLE_W, thickness=2, color=BLUE, spaceAfter=10),
    ]


def _config_section(Q_si: float, r: float, df: pd.DataFrame,
                    method: str, s: dict) -> list:
    t = df["time_s"]
    n_pts = len(df)
    t_min_s, t_max_s = float(t.min()), float(t.max())
    log_cycles = np.log10(t_max_s / t_min_s) if t_min_s > 0 else 0

    # For Papadopulos-Cooper the radial coordinate is the well screen radius,
    # not an observation-well distance — label it accordingly.
    from src.calculator import registry

    spec = registry.get(method)
    at_well = spec is not None and spec.data_source == registry.PUMPED_WELL
    r_label = "Well screen radius r_w" if at_well else "Observation well distance r"
    r_fmt = f"{r:.3f} m" if at_well else f"{r:.1f} m"

    rows = [
        ["Parameter", "Value"],
        ["Pumping rate Q", f"{Q_si:.6f} m³/s  ({Q_si*86400:.2f} m³/day)"],
        [r_label, r_fmt],
        ["Number of data points", str(n_pts)],
        ["Time range", f"{t_min_s:.0f} s  –  {t_max_s:.0f} s"],
        ["Log-cycle span", f"{log_cycles:.2f} log cycles"],
        ["Analysis method", method],
    ]

    flowables = [Paragraph("Test Configuration", s["section"])]
    flowables.append(_make_table(rows, col_widths=[6.5*cm, USABLE_W - 6.5*cm]))
    flowables.append(Spacer(1, 0.3*cm))
    return flowables


def _tracer_config_section(result: CalculationResult, Q_si: float, r: float,
                           df: pd.DataFrame, method: str, s: dict) -> list:
    """The test as a tracer test was run: no drawdown, and a rate only if pumped."""
    from src.calculator import registry

    info = _tracer_info(result)
    t = df["time_s"]
    rows = [["Parameter", "Value"]]
    if registry.require(method).uses_pumping_rate:
        rows.append(["Pumping rate Q", f"{Q_si:.6f} m³/s  ({Q_si*86400:.2f} m³/day)"])
    rows += [
        ["Travel distance", f"{r:.1f} m"],
        ["Injection", info["injection"].capitalize()],
        ["Background concentration", f"{info['background']:g} {info['unit']}"],
        ["Number of data points", str(len(df))],
        ["Time range", f"{float(t.min()) / 3600:.2f} h  –  {float(t.max()) / 3600:.2f} h"],
        ["Analysis method", method],
    ]
    return [
        Paragraph("Test Configuration", s["section"]),
        _make_table(rows, col_widths=[6.5*cm, USABLE_W - 6.5*cm]),
        Spacer(1, 0.3*cm),
    ]


# Readable names for the PDF. ASCII only: reportlab's built-in Helvetica has no
# glyphs for Greek letters, so "alpha" rather than a symbol that renders as a box.
_PARAMETER_NAMES = {
    "T": "Transmissivity T",
    "S": "Storativity S",
    "S_ratio": "Storage ratio S/S'",
    "Sy": "Specific yield Sy",
    "B": "Leakage factor B",
    "r_over_B": "r/B",
    "alpha": "Drainage constant alpha",
    "delay_index": "Delay index 1/alpha",
    "omega": "Storativity ratio omega",
    "lambda": "Interporosity coefficient lambda",
    "S_fracture": "Fracture storativity",
    "S_matrix": "Matrix storativity",
    "v": "Average linear velocity v",
    "t_c": "Advective travel time t_c",
    "alpha_L": "Longitudinal dispersivity alpha_L",
    "D_L": "Dispersion coefficient D_L",
    "Pe": "Peclet number",
    "n_e": "Effective porosity n_e",
    "C0": "Plateau concentration C0",
    "A_over_n": "Mass per flow area M/(A n)",
    "M_over_Q": "Recovered mass over rate M/Q",
    "M_recovered": "Recovered mass",
    "recovery": "Mass recovery fraction",
}


def _pdf_unit(unit: str) -> str:
    return "" if unit in ("", "-") else " " + unit.replace("^2", "²")


def _parameter_row(p) -> list:
    name = _PARAMETER_NAMES.get(p.name, p.name)
    if p.derived:
        name += " (derived)"

    # Table cells take "\n" as a line break within the cell.
    if p.display_unit and p.display_unit != p.unit:
        value = (f"{p.value:.4e}{_pdf_unit(p.unit)}\n"
                 f"({p.display:.2f}{_pdf_unit(p.display_unit)})")
        ci = (f"{p.display_ci[0]:.2f}  –  {p.display_ci[1]:.2f}"
              f"{_pdf_unit(p.display_unit)}" if p.ci else None)
    else:
        value = f"{p.value:.4e}{_pdf_unit(p.unit)}"
        ci = f"{p.ci[0]:.2e}  –  {p.ci[1]:.2e}" if p.ci else None

    if ci is None:
        ci = "derived" if p.derived else ("fixed" if p.fixed else "not available")
    if p.at_bound:
        value += f"\n(at {p.at_bound} bound)"
    if not p.reliable:
        value += "\n(low reliability)"
    return [name, value, ci]


def _results_section(result: CalculationResult, s: dict) -> list:
    """Every estimated and derived parameter. This used to list only T and S,
    which silently dropped S_y and alpha from a Boulton report, B from a
    Hantush-Jacob report, and omega and lambda from a Warren-Root report."""
    rows = [["Parameter", "Value", "95% Confidence Interval"]]
    rows += [_parameter_row(p) for p in result.parameters]
    rows += [
        ["R² (coefficient of determination)", f"{result.r_squared:.6f}", "—"],
        ["RMSE", (f"{result.rmse:.4g} {_tracer_info(result)['unit']}"
                  if _tracer_info(result) else f"{result.rmse*1000:.3f} mm"), "—"],
    ]
    gof = result.gof
    if gof is not None and np.isfinite(gof.aicc):
        rows.append(["AICc (for comparing methods)", f"{gof.aicc:.2f}",
                     f"{gof.n_params} fitted parameters"])

    flowables = [Paragraph("Estimated Parameters", s["section"])]
    flowables.append(_make_table(
        rows,
        col_widths=[5.5*cm, 5.5*cm, USABLE_W - 11.0*cm],
        header_bg=BLUE,
    ))
    unreliable = [p for p in result.parameters if not p.reliable and p.reliability_note]
    for p in unreliable:
        flowables.append(Paragraph(
            f"{_PARAMETER_NAMES.get(p.name, p.name)}: {_sanitise(p.reliability_note)}",
            s["note"]))
    flowables.append(Spacer(1, 0.3*cm))
    return flowables

# Unicode superscripts → ASCII exponent notation. Helvetica (reportlab's
# built-in) has no glyphs for these, so an unmapped superscript renders as a
# black box. The previous version deleted the minus sign but kept the digit,
# which silently turned "10⁻⁵" into "10⁵" — inverting every storativity range
# quoted in the validity notes.
_SUPERSCRIPTS = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁻": "-", "⁺": "+",
}


def _sanitise(text: str) -> str:
    """Escape XML metacharacters and normalise superscripts for reportlab."""
    out = (text
           .replace("&", "&amp;")
           .replace("<", "&lt;")
           .replace(">", "&gt;"))

    # Rewrite runs of superscript characters as ^(...) so the exponent — and
    # crucially its sign — survives into the PDF.
    result_chars = []
    run = []
    for ch in out:
        if ch in _SUPERSCRIPTS:
            run.append(_SUPERSCRIPTS[ch])
            continue
        if run:
            result_chars.append(f"<super>{''.join(run)}</super>")
            run = []
        result_chars.append(ch)
    if run:
        result_chars.append(f"<super>{''.join(run)}</super>")
    return "".join(result_chars)


def _validity_section(result: CalculationResult, s: dict) -> list:
    if not result.validity_notes:
        return []

    flowables = [Paragraph("Validity Assessment", s["section"])]
    for i, note in enumerate(result.validity_notes, 1):
        flowables.append(Paragraph(f"{i}. {_sanitise(note)}", s["note"]))
    flowables.append(Spacer(1, 0.3*cm))
    return flowables


# Figure captions, keyed by registry method id. Kept here rather than in the
# registry because they describe this specific PDF figure, not the method.
_FIGURE_CAPTIONS = {
    "theis_1935": (
        "Theis (1935) log-log type curve. "
        "Blue circles: observed drawdown. Red line: fitted Theis curve."
    ),
    "cooper_jacob_1946": (
        "Cooper-Jacob (1946) semi-log straight line. "
        "Blue circles: valid late-time data (u < 0.02). "
        "Grey crosses: excluded early-time points. Red line: fitted straight line."
    ),
    "papadopulos_cooper_1967": (
        "Papadopulos-Cooper (1967) log-log type curve. "
        "Blue circles: observed drawdown at pumped well. "
        "Red line: fitted PC well function. "
        "Note: S has questionable reliability (paper p. 244)."
    ),
    "ogata_banks_1961": (
        "Ogata-Banks (1961) breakthrough curve. "
        "Blue circles: observed concentration. Red line: fitted solution."
    ),
    "sauty_1980": (
        "Sauty (1980) radial convergent breakthrough curve. "
        "Blue circles: observed concentration. Red line: fitted solution."
    ),
}


def _plot_section(result: CalculationResult, Q_si: float, r: float,
                  method: str, n_excluded: int, s: dict,
                  r_c: float = None) -> list:
    if result.time_s is None or result.drawdown_obs is None:
        return []

    from src.calculator import registry

    spec = registry.get(method)
    if spec is None:
        return []

    # For a method fitted at the pumped well, `r` is the screen radius r_w and
    # the casing radius is a separate input; otherwise `r` is a distance.
    at_well = spec.data_source == registry.PUMPED_WELL
    r_c_used = (r_c if r_c and r_c > 0 else r) if at_well else r_c

    ctx = registry.PlotContext(
        Q=Q_si,
        r=r,
        r_w=r,
        r_c=r_c_used,
        n_excluded=n_excluded,
    )
    fig = (spec.report_plot or _generic_report_plot_fallback)(result, ctx)
    caption = "Figure 1. " + _FIGURE_CAPTIONS.get(
        spec.id,
        f"{spec.label} log-log fit. Blue circles: observed drawdown. "
        "Red line: fitted solution.")
    if at_well and not (r_c and r_c > 0):
        caption += " Casing radius not supplied — curve drawn assuming r_c = r_w."

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", dpi=150, bbox_inches="tight")
    img_buf.seek(0)
    plt.close(fig)

    img = Image(img_buf, width=USABLE_W, height=USABLE_W * 0.55)

    # KeepTogether prevents the heading being orphaned on a different page from the image
    return [
        KeepTogether([
            Paragraph("Diagnostic Plot", s["section"]),
            img,
            Paragraph(caption, s["caption"]),
            Spacer(1, 0.3*cm),
        ])
    ]


def _data_table_section(df: pd.DataFrame, s: dict, tracer: dict = None) -> list:
    MAX_ROWS = 40
    show = df.head(MAX_ROWS).copy()
    show["time_min"] = (show["time_s"] / 60).round(3)

    header = ["Time (s)", "Time (min)",
              f"Concentration ({tracer['unit']})" if tracer else "Drawdown (m)"]
    rows = [header] + [
        [f"{row.time_s:.1f}", f"{row.time_min:.3f}", f"{row.drawdown_m:.4f}"]
        for _, row in show.iterrows()
    ]

    flowables = [
        Paragraph("Field Data", s["section"]),
        Paragraph(
            f"First {min(MAX_ROWS, len(df))} of {len(df)} data points after formatting.",
            s["body"],
        ),
    ]
    flowables.append(_make_table(
        rows,
        col_widths=[4*cm, 4*cm, 4*cm],
        header_bg=DARK_BLUE,
        font_size=8,
    ))
    return flowables


def _project_section(project_info, test_label: str, s: dict) -> list:
    """Client-facing identification: who, where, when. Omitted when no project
    is open, so an ad-hoc analysis still produces a clean report."""
    if project_info is None:
        return []

    fields = [
        ("Project", project_info.title),
        ("Client", project_info.client),
        ("Project number", project_info.project_number),
        ("Location", project_info.location),
        ("Test", test_label),
        ("Test date", project_info.test_date),
        ("Analyst", project_info.analyst),
    ]
    rows = [["Field", "Value"]] + [[k, _sanitise(str(v))] for k, v in fields if v]
    if len(rows) == 1:
        return []

    flowables = [Paragraph("Project", s["section"])]
    flowables.append(_make_table(rows, col_widths=[4.5 * cm, USABLE_W - 4.5 * cm]))
    if project_info.notes:
        flowables.append(Spacer(1, 0.15 * cm))
        flowables.append(Paragraph(_sanitise(project_info.notes), s["body"]))
    flowables.append(Spacer(1, 0.3 * cm))
    return flowables


def _footer_section(method: str, s: dict) -> list:
    """Cites the method actually used. Previously this credited Theis and
    Cooper-Jacob on every report, including reports for other methods."""
    from src.calculator import registry

    spec = registry.get(method)
    citation = spec.citation if spec else method
    return [
        Spacer(1, 0.5*cm),
        HRFlowable(width=USABLE_W, thickness=0.5, color=BORDER),
        Paragraph(f"Method reference: {_sanitise(citation)}", s["caption"]),
        Paragraph("Generated by HydroAssist | BTP, IIT Kharagpur", s["caption"]),
    ]


# ── Table builder ─────────────────────────────────────────────────────────────

def _make_table(
    rows: list,
    col_widths: list = None,
    header_bg: colors.Color = BLUE,
    font_size: int = 9,
) -> Table:
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    n_cols = len(rows[0])
    style = [
        # Header row
        ("BACKGROUND",  (0, 0), (n_cols-1, 0), header_bg),
        ("TEXTCOLOR",   (0, 0), (n_cols-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (n_cols-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (n_cols-1, 0), font_size),
        ("ALIGN",       (0, 0), (n_cols-1, 0), "CENTER"),
        ("TOPPADDING",  (0, 0), (n_cols-1, 0), 6),
        ("BOTTOMPADDING", (0, 0), (n_cols-1, 0), 6),
        # Body rows
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 1), (-1, -1), font_size),
        ("ALIGN",       (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN",       (0, 1), (0, -1),  "LEFT"),
        ("TOPPADDING",  (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        # Alternating row shading
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        # Grid
        ("GRID",        (0, 0), (-1, -1), 0.4, BORDER),
        ("LINEBELOW",   (0, 0), (n_cols-1, 0), 1.5, header_bg),
    ]
    t.setStyle(TableStyle(style))
    return t


# ── Matplotlib diagnostic plots ───────────────────────────────────────────────

def _theis_mpl(result: CalculationResult, Q: float, r: float) -> plt.Figure:
    """Log-log type curve with residuals for Theis."""
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(8, 5.5),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    t = result.time_s
    s_obs = result.drawdown_obs

    # Smooth theoretical curve
    t_s = np.logspace(np.log10(t.min()), np.log10(t.max()), 300)
    u_s = (r**2 * result.S) / (4.0 * result.T * t_s)
    s_s = (Q / (4.0 * np.pi * result.T)) * expn(1, np.maximum(u_s, 1e-10))

    ax_main.scatter(t, s_obs, s=18, color="#1f77b4", facecolors="none",
                    linewidths=1.2, label="Observed", zorder=3)
    ax_main.plot(t_s, s_s, color="#d62728", lw=1.8,
                 label=f"Theis fit  T={result.T_day:.1f} m²/day  S={result.S:.2e}")

    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_ylabel("Drawdown (m)", fontsize=9)
    ax_main.legend(fontsize=8, framealpha=0.9)
    ax_main.grid(True, which="both", color="#e0e0e0", linewidth=0.5)
    ax_main.set_title(
        f"Theis (1935)  |  R² = {result.r_squared:.4f}  |  RMSE = {result.rmse*1000:.2f} mm",
        fontsize=10,
    )

    # Residuals
    if result.drawdown_fitted is not None:
        res = (s_obs - result.drawdown_fitted) * 1000
        ax_res.bar(t, res, color=["#d62728" if v < 0 else "#1f77b4" for v in res],
                   width=t * 0.15, align="center")
        ax_res.axhline(0, color="black", lw=0.8)
        ax_res.set_ylabel("Res. (mm)", fontsize=8)

    ax_res.set_xscale("log")
    ax_res.set_xlabel("Time (s)", fontsize=9)
    ax_res.grid(True, which="both", color="#e0e0e0", linewidth=0.5)

    fig.tight_layout(h_pad=0.5)
    return fig


def _pc_mpl(result: CalculationResult, Q: float,
            r_w: float, r_c: float) -> plt.Figure:
    """Log-log type curve for Papadopulos-Cooper, for PDF embedding.

    Both well radii are passed in explicitly. An earlier version tried to
    recover r_c by string-parsing α out of a validity note; that parse always
    failed and silently fell back to r_c = r_w, so the PDF curve disagreed
    with the on-screen curve whenever r_c ≠ r_w.
    """
    from src.calculator.papadopulos_cooper import pc_drawdown

    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(8, 5.5),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    t = result.time_s
    s_obs = result.drawdown_obs
    alpha = (r_w ** 2 * result.S) / (r_c ** 2)

    t_s = np.logspace(np.log10(t.min()), np.log10(t.max()), 150)
    s_s = pc_drawdown(t_s, result.T, result.S, Q, r_w, r_c)

    ax_main.scatter(t, s_obs, s=18, color="#1f77b4", facecolors="none",
                    linewidths=1.2, label="Observed (pumped well)", zorder=3)
    ax_main.plot(t_s, s_s, color="#d62728", lw=1.8,
                 label=f"PC fit  T={result.T_day:.1f} m²/day  α={alpha:.2e}")

    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_ylabel("Drawdown (m)", fontsize=9)
    ax_main.legend(fontsize=8, framealpha=0.9)
    ax_main.grid(True, which="both", color="#e0e0e0", linewidth=0.5)
    ax_main.set_title(
        f"Papadopulos-Cooper (1967)  |  R² = {result.r_squared:.4f}  |  RMSE = {result.rmse*1000:.2f} mm",
        fontsize=10,
    )

    if result.drawdown_fitted is not None:
        res = (s_obs - result.drawdown_fitted) * 1000
        ax_res.bar(t, res, color=["#d62728" if v < 0 else "#1f77b4" for v in res],
                   width=t * 0.15, align="center")
        ax_res.axhline(0, color="black", lw=0.8)
        ax_res.set_ylabel("Res. (mm)", fontsize=8)

    ax_res.set_xscale("log")
    ax_res.set_xlabel("Time (s)", fontsize=9)
    ax_res.grid(True, which="both", color="#e0e0e0", linewidth=0.5)
    fig.tight_layout(h_pad=0.5)
    return fig


def _cj_mpl(result: CalculationResult, n_excluded: int) -> plt.Figure:
    """Semi-log straight-line plot for Cooper-Jacob."""
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(8, 5.5),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    t = result.time_s
    s_obs = result.drawdown_obs
    s_fit = result.drawdown_fitted

    t_excl = t[:n_excluded]
    s_excl = s_obs[:n_excluded]
    t_late = t[n_excluded:]
    s_late = s_obs[n_excluded:]

    if n_excluded > 0:
        ax_main.scatter(t_excl, s_excl, s=18, color="#aec7e8", marker="x",
                        linewidths=1.2, label=f"Excluded early-time ({n_excluded} pts)", zorder=2)

    ax_main.scatter(t_late, s_late, s=18, color="#1f77b4", facecolors="none",
                    linewidths=1.2, label="Observed (u < 0.02)", zorder=3)

    if s_fit is not None:
        ax_main.plot(t, s_fit, color="#d62728", lw=1.8,
                     label=f"CJ fit  T={result.T_day:.1f} m²/day  S={result.S:.2e}")

    ax_main.set_xscale("log")
    ax_main.set_ylabel("Drawdown (m)", fontsize=9)
    ax_main.legend(fontsize=8, framealpha=0.9)
    ax_main.grid(True, which="both", color="#e0e0e0", linewidth=0.5)
    ax_main.set_title(
        f"Cooper-Jacob (1946)  |  R² = {result.r_squared:.4f}  |  RMSE = {result.rmse*1000:.2f} mm",
        fontsize=10,
    )

    # Residuals on late-time data only
    if s_fit is not None:
        res = (s_obs[n_excluded:] - s_fit[n_excluded:]) * 1000
        ax_res.bar(t_late, res,
                   color=["#d62728" if v < 0 else "#1f77b4" for v in res],
                   width=t_late * 0.15, align="center")
        ax_res.axhline(0, color="black", lw=0.8)
        ax_res.set_ylabel("Res. (mm)", fontsize=8)

    ax_res.set_xscale("log")
    ax_res.set_xlabel("Time (s)", fontsize=9)
    ax_res.grid(True, which="both", color="#e0e0e0", linewidth=0.5)

    fig.tight_layout(h_pad=0.5)
    return fig


# ── Generic report figure, for any method ─────────────────────────────────────

def _recovery_mpl(result: CalculationResult) -> plt.Figure:
    """Residual drawdown against t/t' — the Theis recovery plot."""
    from src.calculator.methods.theis_recovery import plot_axis

    ratio, fitted = plot_axis(result)
    s_obs = np.asarray(result.drawdown_obs, dtype=float)
    s_fit = np.asarray(result.drawdown_fitted, dtype=float)
    slope = result.diagnostics.get("slope_m_per_log_cycle", 0.0)
    ratio_0 = result.value("S_ratio", 1.0)
    t_p_h = result.diagnostics.get("pumping_duration_s", 0.0) / 3600.0

    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(8, 5.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )
    ax_main.scatter(ratio[fitted], s_obs[fitted], s=18, color="#1f77b4",
                    facecolors="none", linewidths=1.2, label="Used in the fit",
                    zorder=3)
    if (~fitted).any():
        ax_main.scatter(ratio[~fitted], s_obs[~fitted], s=14, color="#999999",
                        marker="x", label="Early recovery (not fitted)", zorder=2)
    x_line = np.logspace(0, np.log10(max(float(ratio.max()), 1.0001)), 120)
    ax_main.plot(x_line, slope * (np.log10(x_line) - np.log10(ratio_0)),
                 color="#d62728", lw=1.8,
                 label=f"Recovery line  T={result.T_day:.1f} m²/day  "
                       f"S/S'={ratio_0:.2f}")
    ax_main.axhline(0, color="black", lw=0.6, ls=":")

    res = (s_obs[fitted] - s_fit[fitted]) * 1000
    ax_res.bar(ratio[fitted], res,
               color=["#d62728" if v < 0 else "#1f77b4" for v in res],
               width=ratio[fitted] * 0.04, align="center")
    ax_res.axhline(0, color="black", lw=0.8)
    ax_res.set_ylabel("Res. (mm)", fontsize=8)

    ax_main.set_xscale("log")
    ax_main.set_ylabel("Residual drawdown s' (m)", fontsize=9)
    ax_main.legend(fontsize=8, framealpha=0.9)
    ax_main.grid(True, which="both", color="#e0e0e0", linewidth=0.5)
    ax_main.set_title(
        f"{result.method}  |  t_p = {t_p_h:.2f} h  |  R² = {result.r_squared:.4f}  "
        f"|  RMSE = {result.rmse*1000:.2f} mm", fontsize=10)
    ax_res.set_xscale("log")
    ax_res.set_xlabel("t/t' (time since pumping started / time since it stopped)",
                      fontsize=9)
    ax_res.grid(True, which="both", color="#e0e0e0", linewidth=0.5)
    fig.tight_layout(h_pad=0.5)
    return fig


def _tracer_mpl(result: CalculationResult) -> plt.Figure:
    """The breakthrough curve on linear axes, with residuals in concentration."""
    from src.calculator.methods.tracer import curve_from_result
    from src.calculator.plots import _tracer_fitted

    unit = _tracer_info(result)["unit"]
    t = np.asarray(result.time_s, dtype=float)
    c_obs = np.asarray(result.drawdown_obs, dtype=float)
    c_fit = np.asarray(result.drawdown_fitted, dtype=float)
    smooth = np.linspace(t.min(), t.max(), 300)

    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(8, 5.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )
    ax_main.scatter(t / 3600.0, c_obs, s=18, color="#1f77b4", facecolors="none",
                    linewidths=1.2, label="Observed", zorder=3)
    ax_main.plot(smooth / 3600.0,
                 curve_from_result(result, smooth, *_tracer_fitted(result)),
                 color="#d62728", lw=1.8, label="Fitted solution")
    ax_main.set_ylabel(f"Concentration ({unit})", fontsize=9)
    ax_main.legend(fontsize=8, framealpha=0.9)
    ax_main.grid(True, color="#e0e0e0", linewidth=0.5)
    ax_main.set_title(f"{result.method}  |  R² = {result.r_squared:.4f}  |  "
                      f"RMSE = {result.rmse:.3g} {unit}", fontsize=10)

    ax_res.scatter(t / 3600.0, c_obs - c_fit, s=10, color="#1f77b4")
    ax_res.axhline(0, color="black", lw=0.8)
    ax_res.set_ylabel(f"Res. ({unit})", fontsize=8)
    ax_res.set_xlabel("Time since injection (hours)", fontsize=9)
    ax_res.grid(True, color="#e0e0e0", linewidth=0.5)
    fig.tight_layout(h_pad=0.5)
    return fig


def _generic_report_plot_fallback(result, ctx):
    return _generic_mpl(result)


def _generic_mpl(result: CalculationResult) -> plt.Figure:
    """Log-log observed vs fitted drawdown with residuals, for any method. The
    default report figure, so a newly registered method can always produce a PDF."""
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, figsize=(8, 5.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )
    t = np.asarray(result.time_s, dtype=float)
    s_obs = np.asarray(result.drawdown_obs, dtype=float)
    shown = s_obs > 0

    ax_main.scatter(t[shown], s_obs[shown], s=18, color="#1f77b4",
                    facecolors="none", linewidths=1.2, label="Observed", zorder=3)
    if result.drawdown_fitted is not None:
        s_fit = np.asarray(result.drawdown_fitted, dtype=float)
        order = np.argsort(t)
        keep = s_fit[order] > 0
        ax_main.plot(t[order][keep], s_fit[order][keep], color="#d62728", lw=1.8,
                     label="Fitted solution")
        res = (s_obs - s_fit) * 1000
        ax_res.bar(t, res, color=["#d62728" if v < 0 else "#1f77b4" for v in res],
                   width=t * 0.15, align="center")
        ax_res.axhline(0, color="black", lw=0.8)
        ax_res.set_ylabel("Res. (mm)", fontsize=8)

    ax_main.set_xscale("log")
    ax_main.set_yscale("log")
    ax_main.set_ylabel("Drawdown (m)", fontsize=9)
    ax_main.legend(fontsize=8, framealpha=0.9)
    ax_main.grid(True, which="both", color="#e0e0e0", linewidth=0.5)
    ax_main.set_title(
        f"{result.method}  |  R² = {result.r_squared:.4f}  |  "
        f"RMSE = {result.rmse*1000:.2f} mm", fontsize=10)
    ax_res.set_xscale("log")
    ax_res.set_xlabel("Time (s)", fontsize=9)
    ax_res.grid(True, which="both", color="#e0e0e0", linewidth=0.5)
    fig.tight_layout(h_pad=0.5)
    return fig
