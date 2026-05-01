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
) -> bytes:
    """
    Build a PDF report for a completed pumping test analysis.

    Args:
        result:     CalculationResult from Theis or Cooper-Jacob calculator
        Q_si:       Pumping rate in m³/s
        r:          Observation well distance in m
        method:     Method label string ("Theis (1935)" or "Cooper-Jacob (1946)")
        df:         Formatted DataFrame (time_s, drawdown_m)
        n_excluded: Number of early-time points excluded (Cooper-Jacob only)

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
    story += _config_section(Q_si, r, df, method, styles)
    story += _results_section(result, styles)
    story += _validity_section(result, styles)
    story += _plot_section(result, Q_si, r, method, n_excluded, styles)
    story += _data_table_section(df, styles)
    story += _footer_section(styles)

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

def _header_section(method: str, s: dict) -> list:
    now = datetime.now().strftime("%Y-%m-%d  %H:%M")
    return [
        Paragraph("HydroAssist", s["title"]),
        Paragraph("Confined Aquifer Pumping Test Analysis", s["subtitle"]),
        Paragraph(f"Method: <b>{method}</b>  |  Generated: {now}", s["subtitle"]),
        HRFlowable(width=USABLE_W, thickness=2, color=BLUE, spaceAfter=10),
    ]


def _config_section(Q_si: float, r: float, df: pd.DataFrame,
                    method: str, s: dict) -> list:
    t = df["time_s"]
    n_pts = len(df)
    t_min_s, t_max_s = float(t.min()), float(t.max())
    log_cycles = np.log10(t_max_s / t_min_s) if t_min_s > 0 else 0

    rows = [
        ["Parameter", "Value"],
        ["Pumping rate Q", f"{Q_si:.6f} m³/s  ({Q_si*86400:.2f} m³/day)"],
        ["Observation well distance r", f"{r:.1f} m"],
        ["Number of data points", str(n_pts)],
        ["Time range", f"{t_min_s:.0f} s  –  {t_max_s:.0f} s"],
        ["Log-cycle span", f"{log_cycles:.2f} log cycles"],
        ["Analysis method", method],
    ]

    flowables = [Paragraph("Test Configuration", s["section"])]
    flowables.append(_make_table(rows, col_widths=[6.5*cm, USABLE_W - 6.5*cm]))
    flowables.append(Spacer(1, 0.3*cm))
    return flowables


def _results_section(result: CalculationResult, s: dict) -> list:
    T_ci_str = (
        f"{result.T_ci[0]*86400:.2f}  –  {result.T_ci[1]*86400:.2f} m²/day"
        if result.T_ci else "not available"
    )
    S_ci_str = (
        f"{result.S_ci[0]:.2e}  –  {result.S_ci[1]:.2e}"
        if result.S_ci else "not available"
    )

    rows = [
        ["Parameter", "Value", "95% Confidence Interval"],
        ["Transmissivity T",
         f"{result.T:.4e} m²/s\n({result.T_day:.2f} m²/day)",
         T_ci_str],
        ["Storativity S",
         f"{result.S:.4e}",
         S_ci_str],
        ["R² (coefficient of determination)", f"{result.r_squared:.6f}", "—"],
        ["RMSE", f"{result.rmse*1000:.3f} mm", "—"],
    ]

    flowables = [Paragraph("Estimated Parameters", s["section"])]
    flowables.append(_make_table(
        rows,
        col_widths=[5.5*cm, 5.5*cm, USABLE_W - 11.0*cm],
        header_bg=BLUE,
    ))
    flowables.append(Spacer(1, 0.3*cm))
    return flowables


def _validity_section(result: CalculationResult, s: dict) -> list:
    if not result.validity_notes:
        return []

    flowables = [Paragraph("Validity Assessment", s["section"])]
    for i, note in enumerate(result.validity_notes, 1):
        # Sanitise special characters for reportlab XML parser
        clean = (note
                 .replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;")
                 .replace("²", "&#178;")
                 .replace("⁰", "")
                 .replace("⁻", "")
                 .replace("³", "&#179;"))
        flowables.append(Paragraph(f"{i}. {clean}", s["note"]))
    flowables.append(Spacer(1, 0.3*cm))
    return flowables


def _plot_section(result: CalculationResult, Q_si: float, r: float,
                  method: str, n_excluded: int, s: dict) -> list:
    if result.time_s is None or result.drawdown_obs is None:
        return []

    if "Theis" in method:
        fig = _theis_mpl(result, Q_si, r)
        caption = (
            "Figure 1. Theis (1935) log-log type curve. "
            "Blue circles: observed drawdown. Red line: fitted Theis curve."
        )
    elif "Cooper-Jacob" in method:
        fig = _cj_mpl(result, n_excluded)
        caption = (
            "Figure 1. Cooper-Jacob (1946) semi-log straight line. "
            "Blue circles: valid late-time data (u < 0.02). "
            "Grey crosses: excluded early-time points. Red line: fitted straight line."
        )
    else:
        # Papadopulos-Cooper — r is r_w here (passed as such from app.py)
        fig = _pc_mpl(result, Q_si, r)
        caption = (
            "Figure 1. Papadopulos-Cooper (1967) log-log type curve. "
            "Blue circles: observed drawdown at pumped well. "
            "Red line: fitted PC well function. "
            "Note: S has questionable reliability (paper p. 244)."
        )

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


def _data_table_section(df: pd.DataFrame, s: dict) -> list:
    MAX_ROWS = 40
    show = df.head(MAX_ROWS).copy()
    show["time_min"] = (show["time_s"] / 60).round(3)

    header = ["Time (s)", "Time (min)", "Drawdown (m)"]
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


def _footer_section(s: dict) -> list:
    return [
        Spacer(1, 0.5*cm),
        HRFlowable(width=USABLE_W, thickness=0.5, color=BORDER),
        Paragraph(
            "Generated by HydroAssist | BTP-2, IIT Kharagpur | "
            "Theis (1935) and Cooper &amp; Jacob (1946) — confined aquifer methods.",
            s["caption"],
        ),
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


def _pc_mpl(result: CalculationResult, Q: float, r_w: float) -> plt.Figure:
    """Log-log type curve for Papadopulos-Cooper, for PDF embedding."""
    from src.calculator.papadopulos_cooper import pc_drawdown

    # r_w is passed as `r` from the caller (app.py sets r = r_w for PC)
    # We need r_c too; derive from α and S if possible, else assume r_c = r_w
    alpha_fit = None
    for note in result.validity_notes:
        if "α =" in note:
            try:
                part = note.split("α =")[1].strip().split()[0].rstrip(".")
                alpha_fit = float(part)
            except (IndexError, ValueError):
                pass
    r_c = r_w  # default: r_c = r_w
    if alpha_fit is not None and result.S > 0:
        r_c_sq = r_w ** 2 * result.S / alpha_fit
        if r_c_sq > 0:
            r_c = float(np.sqrt(r_c_sq))

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
