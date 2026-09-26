"""Diagnostic plots for pumping test analysis.

Two plot types per method:
  - Static diagnostic plot  : standard hydrogeological display of the fit result
  - Interactive manual plot : sliders let the analyst adjust T/S and see the
                              theoretical curve update in real time

All functions return plotly.graph_objects.Figure objects for Streamlit display.
"""

import numpy as np
from scipy.special import expn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.calculator.base import CalculationResult
from src.calculator.papadopulos_cooper import pc_drawdown

# ── Colour palette (consistent across all plots) ──────────────────────────────
COL_OBS    = "#1f77b4"   # blue   — observed data
COL_FIT    = "#d62728"   # red    — fitted / theoretical curve
COL_EXCL   = "#aec7e8"   # light blue — excluded early-time points (CJ)
COL_MANUAL = "#ff7f0e"   # orange — manual slider curve
COL_GRID   = "#e0e0e0"

# Plausible S range for the Papadopulos-Cooper reference type curves.
# Defined here (not at the bottom of the module) so pc_diagnostic's reference
# curve guard reads in source order.
S_MIN_PLOT = 1e-10
S_MAX_PLOT = 1e-1

# Horizontal legend centred below the figure — keeps the plot area clear
_LEGEND_H = dict(
    orientation="h",
    yanchor="top",
    y=-0.18,
    xanchor="center",
    x=0.5,
    bgcolor="rgba(0,0,0,0)",
    bordercolor="rgba(0,0,0,0)",
)

# Extra bottom margin so the legend has room
_MARGIN_DUAL  = dict(t=55, b=110, l=60, r=20)   # two-panel static plots
_MARGIN_SINGLE = dict(t=55, b=100, l=60, r=20)  # single-panel interactive plots

# ── The two-panel frame: a fit above, its residuals below ─────────────────────
#
# Written once because it was written five times, and the five had drifted into
# three faults that a reviewer would notice before they noticed the science:
#
# - The top panel carried an x-axis title of its own even though the axes are
#   shared, so "Time (s)" was drawn in the gap between the panels, on top of the
#   "Residuals" heading. With shared axes only the bottom panel is labelled.
# - The zero line was white. On the light theme the app actually uses, the one
#   line a residual plot is read against was invisible.
# - Residuals were bars on a logarithmic time axis, where a bar's width means
#   nothing and neighbouring bars at late time overlap into a block. Markers on a
#   zero line are how a residual plot is read, and the ±RMSE band gives the eye
#   the scale that the numbers in the title state.

# Enough room under the top panel for its x tick labels and the lower heading.
_PANEL_SPACING = 0.13
_PANEL_HEIGHTS = [0.72, 0.28]

COL_ZERO = "#6c757d"   # grey — the zero line, and the RMSE band's edges


def _two_panel(top_title: str, bottom_title: str = "Residuals") -> go.Figure:
    """A fit above and its residuals below, sharing one time axis."""
    return make_subplots(
        rows=2, cols=1,
        row_heights=_PANEL_HEIGHTS,
        shared_xaxes=True,
        vertical_spacing=_PANEL_SPACING,
        subplot_titles=(top_title, bottom_title),
    )


def _add_residuals(fig, x, residual_m, rmse_m: float = 0.0) -> None:
    """The lower panel: residuals in millimetres about zero.

    Markers rather than bars, a visible zero line, and a shaded ±RMSE band so
    the scatter can be read against the fit statistic quoted in the title.
    """
    residual_mm = np.asarray(residual_m) * 1000.0

    # The trace goes first, and not for tidiness: `add_hline(row=…)` defaults to
    # `exclude_empty_subplots=True`, and a panel with no trace in it yet counts as
    # empty — so a zero line added before its points is silently dropped, and the
    # figure comes back looking right apart from the one line the panel is read
    # against. Pinned by test_the_zero_line_is_visible_on_a_white_page.
    fig.add_trace(go.Scatter(
        x=x, y=residual_mm, mode="markers", name="Residual (mm)",
        marker=dict(color=COL_OBS, size=6, symbol="circle",
                    line=dict(width=1, color=COL_FIT)),
        hovertemplate="t = %{x:.1f} s<br>residual = %{y:.2f} mm<extra></extra>",
        showlegend=False,
    ), row=2, col=1)

    if rmse_m > 0:
        fig.add_hrect(
            y0=-rmse_m * 1000.0, y1=rmse_m * 1000.0,
            line_width=0, fillcolor=COL_OBS, opacity=0.10, layer="below",
            row=2, col=1,
        )
    fig.add_hline(y=0, line=dict(color=COL_ZERO, width=1), layer="below",
                  row=2, col=1)


def _label_panels(fig, x_title: str, y_title: str, y_log: bool = True,
                  residual_title: str = "Residual (mm)") -> None:
    """Axis titles in the only places they belong on a shared-axis pair."""
    # Top panel: no x title — the axis is shared and labelled once, below.
    fig.update_xaxes(type="log", showgrid=True, gridcolor=COL_GRID, row=1, col=1)
    fig.update_yaxes(type="log" if y_log else "linear", title_text=y_title,
                     showgrid=True, gridcolor=COL_GRID, row=1, col=1)
    fig.update_xaxes(type="log", title_text=x_title, showgrid=True,
                     gridcolor=COL_GRID, row=2, col=1)
    fig.update_yaxes(title_text=residual_title, showgrid=True,
                     gridcolor=COL_GRID, zeroline=False, row=2, col=1)


# ── Theis static plot ─────────────────────────────────────────────────────────

def theis_diagnostic(result: CalculationResult, Q: float, r: float) -> go.Figure:
    """Log-log type curve + residuals for a Theis fit.

    Q [m³/s] and r [m] are the test parameters the fit was run with. They are
    not recoverable from CalculationResult, so the caller must supply them —
    without them the theoretical curve cannot be drawn at all.
    """
    fig = _two_panel(
        f"Theis (1935) — Log-Log Type Curve   "
        f"(R² = {result.r_squared:.4f}, RMSE = {result.rmse*1000:.2f} mm)")

    t      = result.time_s
    s_obs  = result.drawdown_obs
    s_fit  = result.drawdown_fitted

    fig.add_trace(go.Scatter(
        x=t, y=s_obs, mode="markers", name="Observed",
        marker=dict(color=COL_OBS, size=7, symbol="circle-open", line=dict(width=1.5)),
        hovertemplate="t = %{x:.1f} s<br>s = %{y:.4f} m<extra></extra>",
    ), row=1, col=1)

    t_smooth = np.logspace(np.log10(t.min()), np.log10(t.max()), 300)
    u_s = (r**2 * result.S) / (4.0 * result.T * t_smooth)
    s_s = (Q / (4.0 * np.pi * result.T)) * expn(1, np.maximum(u_s, 1e-10))
    fig.add_trace(go.Scatter(
        x=t_smooth, y=s_s, mode="lines",
        name=f"Theis fit  (T = {result.T_day:.1f} m²/day, S = {result.S:.2e})",
        line=dict(color=COL_FIT, width=2),
        hovertemplate="t = %{x:.1f} s<br>s_fit = %{y:.4f} m<extra></extra>",
    ), row=1, col=1)

    if s_fit is not None:
        _add_residuals(fig, t, s_obs - s_fit, result.rmse)

    _label_panels(fig, "Time (s)", "Drawdown (m)")
    fig.update_layout(height=560, legend=_LEGEND_H, margin=_MARGIN_DUAL, hovermode="x unified")
    return fig


# ── Cooper-Jacob static plot ──────────────────────────────────────────────────

def cj_diagnostic(result: CalculationResult, Q: float, r: float,
                  n_excluded: int = 0) -> go.Figure:
    """Semi-log straight line + residuals for a Cooper-Jacob fit.

    Q [m³/s] and r [m] are the test parameters the fit was run with; r is
    needed to place the t₀ intercept marker (paper Eq. 9).
    """
    fig = _two_panel(
        f"Cooper-Jacob (1946) — Semi-Log Straight Line   "
        f"(R² = {result.r_squared:.4f}, RMSE = {result.rmse*1000:.2f} mm)",
        "Residuals (the late-time readings the line was fitted to)")

    t     = result.time_s
    s_obs = result.drawdown_obs
    s_fit = result.drawdown_fitted

    t_excl, s_excl = t[:n_excluded], s_obs[:n_excluded]
    t_late, s_late = t[n_excluded:], s_obs[n_excluded:]

    if n_excluded > 0:
        fig.add_trace(go.Scatter(
            x=t_excl, y=s_excl, mode="markers",
            name=f"Excluded ({n_excluded} pts, u ≥ 0.02)",
            marker=dict(color=COL_EXCL, size=7, symbol="circle-open", line=dict(width=1.5)),
            hovertemplate="t = %{x:.1f} s<br>s = %{y:.4f} m<extra></extra>",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t_late, y=s_late, mode="markers", name="Observed (u < 0.02)",
        marker=dict(color=COL_OBS, size=7, symbol="circle-open", line=dict(width=1.5)),
        hovertemplate="t = %{x:.1f} s<br>s = %{y:.4f} m<extra></extra>",
    ), row=1, col=1)

    if s_fit is not None:
        fig.add_trace(go.Scatter(
            x=t, y=s_fit, mode="lines",
            name=f"CJ fit  (T = {result.T_day:.1f} m²/day, S = {result.S:.2e})",
            line=dict(color=COL_FIT, width=2),
            hovertemplate="t = %{x:.1f} s<br>s_fit = %{y:.4f} m<extra></extra>",
        ), row=1, col=1)

    # t₀ — zero-drawdown intercept, inverted from paper Eq. 9: S = 2.25·T·t₀/r²
    t0 = compute_t0(result, r)
    if t0 and t0 > 0:
        fig.add_vline(
            x=np.log10(t0), line=dict(color="grey", dash="dash", width=1),
            annotation_text=f"t₀ = {t0:.1f} s", annotation_position="top right",
            row=1, col=1,
        )

    if s_fit is not None:
        # Only the points the line was fitted to: a residual against a straight
        # line that deliberately ignores early time is not a misfit.
        _add_residuals(fig, t_late, s_obs[n_excluded:] - s_fit[n_excluded:],
                       result.rmse)

    _label_panels(fig, "Time (s)", "Drawdown (m)", y_log=False)
    fig.update_layout(height=560, legend=_LEGEND_H, margin=_MARGIN_DUAL, hovermode="x unified")
    return fig


# ── Theis interactive (manual matching) ───────────────────────────────────────

def theis_interactive(
    time_s, drawdown_obs, Q, r,
    T_fit, S_fit, T_manual, S_manual,
) -> tuple:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_s, y=drawdown_obs, mode="markers", name="Observed",
        marker=dict(color=COL_OBS, size=7, symbol="circle-open", line=dict(width=1.5)),
        hovertemplate="t = %{x:.1f} s<br>s = %{y:.4f} m<extra></extra>",
    ))

    t_smooth = np.logspace(np.log10(time_s.min()), np.log10(time_s.max()), 300)

    u_fit = (r**2 * S_fit) / (4.0 * T_fit * t_smooth)
    s_auto = (Q / (4.0 * np.pi * T_fit)) * expn(1, np.maximum(u_fit, 1e-10))
    fig.add_trace(go.Scatter(
        x=t_smooth, y=s_auto, mode="lines",
        name=f"Auto fit  (T = {T_fit*86400:.1f} m²/day, S = {S_fit:.2e})",
        line=dict(color=COL_FIT, width=2, dash="dot"),
    ))

    u_man = (r**2 * S_manual) / (4.0 * T_manual * t_smooth)
    s_man = (Q / (4.0 * np.pi * T_manual)) * expn(1, np.maximum(u_man, 1e-10))
    fig.add_trace(go.Scatter(
        x=t_smooth, y=s_man, mode="lines",
        name=f"Manual  (T = {T_manual*86400:.1f} m²/day, S = {S_manual:.2e})",
        line=dict(color=COL_MANUAL, width=2.5),
    ))

    u_obs = (r**2 * S_manual) / (4.0 * T_manual * time_s)
    s_man_obs = (Q / (4.0 * np.pi * T_manual)) * expn(1, np.maximum(u_obs, 1e-10))
    rmse_mm = float(np.sqrt(np.mean((drawdown_obs - s_man_obs)**2))) * 1000

    fig.update_xaxes(type="log", title_text="Time (s)", showgrid=True, gridcolor=COL_GRID)
    fig.update_yaxes(type="log", title_text="Drawdown (m)", showgrid=True, gridcolor=COL_GRID)
    fig.update_layout(
        title=f"Theis Manual Matching — RMSE = {rmse_mm:.2f} mm",
        height=480, legend=_LEGEND_H, hovermode="x unified", margin=_MARGIN_SINGLE,
    )
    return fig, rmse_mm


# ── Cooper-Jacob interactive (manual matching) ────────────────────────────────

def cj_interactive(
    time_s, drawdown_obs, Q, r,
    T_fit, S_fit, T_manual, S_manual,
) -> tuple:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_s, y=drawdown_obs, mode="markers", name="Observed",
        marker=dict(color=COL_OBS, size=7, symbol="circle-open", line=dict(width=1.5)),
        hovertemplate="t = %{x:.1f} s<br>s = %{y:.4f} m<extra></extra>",
    ))

    t_smooth = np.logspace(np.log10(time_s.min()), np.log10(time_s.max()), 300)

    slope_fit = (2.303 * Q) / (4.0 * np.pi * T_fit)
    t0_fit = (S_fit * r**2) / (2.25 * T_fit)
    s_auto = slope_fit * np.log10(t_smooth) + (-slope_fit * np.log10(t0_fit))
    fig.add_trace(go.Scatter(
        x=t_smooth, y=s_auto, mode="lines",
        name=f"Auto fit  (T = {T_fit*86400:.1f} m²/day, S = {S_fit:.2e})",
        line=dict(color=COL_FIT, width=2, dash="dot"),
    ))

    slope_man = (2.303 * Q) / (4.0 * np.pi * T_manual)
    t0_man = (S_manual * r**2) / (2.25 * T_manual)
    s_man = slope_man * np.log10(t_smooth) + (-slope_man * np.log10(t0_man))
    fig.add_trace(go.Scatter(
        x=t_smooth, y=s_man, mode="lines",
        name=f"Manual  (T = {T_manual*86400:.1f} m²/day, S = {S_manual:.2e})",
        line=dict(color=COL_MANUAL, width=2.5),
    ))

    if t0_man > 0:
        fig.add_vline(
            x=np.log10(t0_man), line=dict(color=COL_MANUAL, dash="dash", width=1),
            annotation_text=f"t₀ = {t0_man:.1f} s", annotation_position="top left",
        )

    s_man_obs = slope_man * np.log10(time_s) + (-slope_man * np.log10(t0_man))
    rmse_mm = float(np.sqrt(np.mean((drawdown_obs - s_man_obs)**2))) * 1000

    fig.update_xaxes(type="log", title_text="Time (s)", showgrid=True, gridcolor=COL_GRID)
    fig.update_yaxes(title_text="Drawdown (m)", showgrid=True, gridcolor=COL_GRID)
    fig.update_layout(
        title=f"Cooper-Jacob Manual Matching — RMSE = {rmse_mm:.2f} mm",
        height=480, legend=_LEGEND_H, hovermode="x unified", margin=_MARGIN_SINGLE,
    )
    return fig, rmse_mm


# ── Papadopulos-Cooper static diagnostic plot ─────────────────────────────────

def pc_diagnostic(result: CalculationResult, Q: float,
                  r_w: float, r_c: float) -> go.Figure:
    alpha_fit = (r_w ** 2 * result.S) / (r_c ** 2)

    fig = _two_panel(
        f"Papadopulos-Cooper (1967) — Log-Log Type Curve  (α = {alpha_fit:.2e})")

    t     = result.time_s
    s_obs = result.drawdown_obs
    s_fit = result.drawdown_fitted

    t_smooth = np.logspace(np.log10(t.min()), np.log10(t.max()), 120)

    # Reference type curves for α ×10 and α /10
    for alpha_ref, label, dash in [
        (alpha_fit * 10, f"α×10 = {alpha_fit*10:.1e}", "dot"),
        (alpha_fit / 10, f"α/10 = {alpha_fit/10:.1e}", "dot"),
    ]:
        if 1e-8 < alpha_ref < 1.0:
            S_ref = alpha_ref * r_c ** 2 / r_w ** 2
            if S_MIN_PLOT < S_ref < S_MAX_PLOT:
                s_ref = pc_drawdown(t_smooth, result.T, S_ref, Q, r_w, r_c)
                fig.add_trace(go.Scatter(
                    x=t_smooth, y=s_ref, mode="lines", name=label,
                    line=dict(color="#888888", width=1.2, dash=dash),
                    hoverinfo="skip",
                ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t, y=s_obs, mode="markers", name="Observed (pumped well)",
        marker=dict(color=COL_OBS, size=7, symbol="circle-open", line=dict(width=1.5)),
        hovertemplate="t = %{x:.1f} s<br>s = %{y:.4f} m<extra></extra>",
    ), row=1, col=1)

    s_fitted_smooth = pc_drawdown(t_smooth, result.T, result.S, Q, r_w, r_c)
    fig.add_trace(go.Scatter(
        x=t_smooth, y=s_fitted_smooth, mode="lines", name="PC fit",
        line=dict(color=COL_FIT, width=2),
    ), row=1, col=1)

    if s_fit is not None:
        res = s_obs - s_fit
        _add_residuals(fig, t, res, result.rmse)

    # Shade wellbore-storage zone
    u_w_arr = (r_w ** 2 * result.S) / (4.0 * result.T * t_smooth)
    ws_mask = u_w_arr / alpha_fit > 1.0
    if ws_mask.any():
        t_ws_end = float(t_smooth[ws_mask][-1])
        fig.add_vrect(
            x0=np.log10(t.min()), x1=np.log10(t_ws_end),
            fillcolor="rgba(255,200,100,0.10)", line_width=0,
            annotation_text="Wellbore storage", annotation_position="top left",
            row=1, col=1,
        )

    _label_panels(fig, "Time (s)", "Drawdown (m)")
    fig.update_layout(height=580, legend=_LEGEND_H, margin=_MARGIN_DUAL, hovermode="x unified")
    return fig


# ── Papadopulos-Cooper interactive (manual matching) ──────────────────────────

def pc_interactive(
    time_s, drawdown_obs, Q, r_w, r_c,
    T_fit, S_fit, T_manual, S_manual,
) -> tuple:
    fig = go.Figure()

    t_smooth = np.logspace(np.log10(time_s.min()), np.log10(time_s.max()), 60)

    fig.add_trace(go.Scatter(
        x=time_s, y=drawdown_obs, mode="markers", name="Observed",
        marker=dict(color=COL_OBS, size=7, symbol="circle-open", line=dict(width=1.5)),
        hovertemplate="t = %{x:.1f} s<br>s = %{y:.4f} m<extra></extra>",
    ))

    s_auto = pc_drawdown(t_smooth, T_fit, S_fit, Q, r_w, r_c)
    alpha_fit = r_w ** 2 * S_fit / r_c ** 2
    fig.add_trace(go.Scatter(
        x=t_smooth, y=s_auto, mode="lines",
        name=f"Auto fit  (T = {T_fit*86400:.1f} m²/day, α = {alpha_fit:.2e})",
        line=dict(color=COL_FIT, width=2, dash="dot"),
    ))

    s_man = pc_drawdown(t_smooth, T_manual, S_manual, Q, r_w, r_c)
    alpha_man = r_w ** 2 * S_manual / r_c ** 2
    fig.add_trace(go.Scatter(
        x=t_smooth, y=s_man, mode="lines",
        name=f"Manual  (T = {T_manual*86400:.1f} m²/day, α = {alpha_man:.2e})",
        line=dict(color=COL_MANUAL, width=2.5),
    ))

    s_man_obs = pc_drawdown(time_s, T_manual, S_manual, Q, r_w, r_c)
    rmse_mm = float(np.sqrt(np.mean((drawdown_obs - s_man_obs) ** 2))) * 1000

    fig.update_xaxes(type="log", title_text="Time (s)", showgrid=True, gridcolor=COL_GRID)
    fig.update_yaxes(type="log", title_text="Drawdown (m)", showgrid=True, gridcolor=COL_GRID)
    fig.update_layout(
        title=f"PC Manual Matching — RMSE = {rmse_mm:.2f} mm",
        height=480, legend=_LEGEND_H, hovermode="x unified", margin=_MARGIN_SINGLE,
    )
    return fig, rmse_mm


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_t0(result: CalculationResult, r: float):
    """Zero-drawdown time intercept t₀ [s] for a Cooper-Jacob fit.

    Inverts Cooper & Jacob (1946) equation 9:
        S = 2.25 · T · t₀ / r²   →   t₀ = S · r² / (2.25 · T)

    Returns None when T, S or r are non-physical (a failed fit), so callers
    can simply skip the marker.
    """
    if result.T > 0 and result.S > 0 and r > 0:
        return (result.S * r ** 2) / (2.25 * result.T)
    return None


# ── Bourdet derivative diagnostic ─────────────────────────────────────────────

def derivative_diagnostic(time_s, drawdown_m, Q=None, window=None) -> go.Figure:
    """Log-log drawdown with its Bourdet derivative overlaid.

    The standard modern well-test diagnostic, and the plot a hydrogeologist looks
    at *first*. Drawdown and derivative share one axis deliberately: where the two
    curves lie on top of each other, all the water is coming from wellbore storage,
    which is the clearest visual signature on the whole plot.

    A flat derivative marks infinite-acting radial flow, and its height is
    Q/(4πT) — so T can be read straight off this plot without fitting anything.
    That reference line is drawn whenever Q is known.
    """
    from src.calculator.derivative import DEFAULT_WINDOW, diagnose, log_derivative

    t = np.asarray(time_s, dtype=float)
    s = np.asarray(drawdown_m, dtype=float)
    window = DEFAULT_WINDOW if window is None else window

    d = log_derivative(t, s, window)
    signature = diagnose(t, s, Q=Q, window=window)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t, y=s, mode="markers", name="Drawdown",
        marker=dict(color=COL_OBS, size=7, symbol="circle-open",
                    line=dict(width=1.5)),
        hovertemplate="t = %{x:.1f} s<br>s = %{y:.4f} m<extra></extra>",
    ))

    # Only positive values can be shown on a log axis; a negative derivative is
    # noise in a falling section, not signal.
    positive = d > 0
    fig.add_trace(go.Scatter(
        x=t[positive], y=d[positive], mode="markers+lines",
        name="Derivative  ds/d ln t",
        marker=dict(color=COL_MANUAL, size=6, symbol="x"),
        line=dict(color=COL_MANUAL, width=1.5),
        hovertemplate="t = %{x:.1f} s<br>ds/dlnt = %{y:.4f} m<extra></extra>",
    ))

    if signature.implied_T:
        level = Q / (4.0 * np.pi * signature.implied_T)
        fig.add_hline(
            y=level, line=dict(color=COL_FIT, dash="dash", width=1.5),
            annotation_text=(
                f"radial plateau → T = {signature.implied_T * 86400:.1f} m²/day"
            ),
            annotation_position="bottom right",
        )

    fig.update_xaxes(type="log", title_text="Time (s)",
                     showgrid=True, gridcolor=COL_GRID)
    fig.update_yaxes(type="log", title_text="Drawdown and derivative (m)",
                     showgrid=True, gridcolor=COL_GRID)
    fig.update_layout(
        title=f"Diagnostic plot — {signature.regime.replace('_', ' ')}",
        height=520, legend=_LEGEND_H, hovermode="x unified",
        margin=_MARGIN_SINGLE,
    )
    return fig


# ── Generic plots, for any method ─────────────────────────────────────────────
# Every CalculationResult carries its observed and fitted series and its full
# parameter list, which is enough to draw a correct diagnostic for any method.
# These are the registry's DEFAULT adapters, so a newly registered method gets
# working plots without having to supply its own. Hantush-Jacob, Boulton and
# Warren-Root originally shipped without adapters, and their results panels
# crashed after a successful fit; that is the failure these exist to prevent.

def _parameter_summary(result: CalculationResult) -> str:
    """Estimated parameters as one legend-friendly line."""
    parts = []
    for p in result.parameters:
        if not p.estimated:
            continue
        if p.display_unit and p.display_unit != p.unit:
            unit = p.display_unit.replace("^2", "²")
            parts.append(f"{p.symbol} = {p.display:.1f} {unit}")
        else:
            unit = "" if p.unit in ("", "-") else " " + p.unit.replace("^2", "²")
            parts.append(f"{p.symbol} = {p.value:.3g}{unit}")
    return ", ".join(parts)


def generic_diagnostic(result: CalculationResult, title: str = None) -> go.Figure:
    """Log-log observed vs fitted drawdown, with residuals, for any method."""
    t = np.asarray(result.time_s, dtype=float)
    s_obs = np.asarray(result.drawdown_obs, dtype=float)
    s_fit = (np.asarray(result.drawdown_fitted, dtype=float)
             if result.drawdown_fitted is not None else None)

    fig = _two_panel(
        f"{title or result.method} — Log-Log Fit   "
        f"(R² = {result.r_squared:.4f}, RMSE = {result.rmse * 1000:.2f} mm)")

    shown = s_obs > 0          # a log axis cannot place non-positive values
    fig.add_trace(go.Scatter(
        x=t[shown], y=s_obs[shown], mode="markers", name="Observed",
        marker=dict(color=COL_OBS, size=7, symbol="circle-open", line=dict(width=1.5)),
        hovertemplate="t = %{x:.1f} s<br>s = %{y:.4f} m<extra></extra>",
    ), row=1, col=1)

    if s_fit is not None:
        order = np.argsort(t)
        keep = s_fit[order] > 0
        fig.add_trace(go.Scatter(
            x=t[order][keep], y=s_fit[order][keep], mode="lines",
            name=f"Fit  ({_parameter_summary(result)})",
            line=dict(color=COL_FIT, width=2),
            hovertemplate="t = %{x:.1f} s<br>s_fit = %{y:.4f} m<extra></extra>",
        ), row=1, col=1)

        res = s_obs - s_fit
        _add_residuals(fig, t, res, result.rmse)

    _label_panels(fig, "Time (s)", "Drawdown (m)")
    fig.update_layout(height=560, legend=_LEGEND_H, margin=_MARGIN_DUAL,
                      hovermode="x unified")
    return fig


def template_forward(spec, result: CalculationResult, ctx):
    """forward(t, T, S) -> drawdown for a template-based method, holding every
    parameter other than T and S at its fitted value.

    That is what lets the manual-matching sliders (which move T and S) work for a
    method with four parameters. Returns None for methods not built on
    CurveFitCalculator, whose plots are supplied explicitly anyway.
    """
    import pandas as pd

    from src.calculator.base import CalculationInput
    from src.calculator.templates import CurveFitCalculator

    calc = spec.build()
    if not isinstance(calc, CurveFitCalculator) or result.time_s is None:
        return None

    at_well = spec.data_source == "pumped_well"
    inputs = CalculationInput(
        df=pd.DataFrame({"time_s": result.time_s, "drawdown_m": result.drawdown_obs}),
        Q=ctx.Q, r=ctx.r_w if at_well else ctx.r, r_w=ctx.r_w, r_c=ctx.r_c,
    )
    names = [p.name for p in calc.parameter_specs(
        inputs, np.asarray(result.time_s), np.asarray(result.drawdown_obs))]
    model = calc.model(inputs)
    fitted = {n: result.value(n) for n in names}

    def forward(t, T, S):
        values = dict(fitted, T=T, S=S)
        return model(np.asarray(t, dtype=float), *[values[n] for n in names])

    return forward


def generic_interactive(result: CalculationResult, forward,
                        T_manual: float, S_manual: float,
                        title: str = None) -> tuple:
    """Manual matching for any method. With `forward` None, shows the automatic
    fit only and reports its RMSE."""
    t = np.asarray(result.time_s, dtype=float)
    s_obs = np.asarray(result.drawdown_obs, dtype=float)
    t_smooth = np.logspace(np.log10(t.min()), np.log10(t.max()), 120)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=s_obs, mode="markers", name="Observed",
        marker=dict(color=COL_OBS, size=7, symbol="circle-open", line=dict(width=1.5)),
    ))

    if forward is None:
        order = np.argsort(t)
        fig.add_trace(go.Scatter(
            x=t[order], y=np.asarray(result.drawdown_fitted)[order], mode="lines",
            name="Auto fit", line=dict(color=COL_FIT, width=2, dash="dot"),
        ))
        rmse_mm = float(result.rmse * 1000)
    else:
        fig.add_trace(go.Scatter(
            x=t_smooth, y=forward(t_smooth, result.T, result.S), mode="lines",
            name=f"Auto fit  ({_parameter_summary(result)})",
            line=dict(color=COL_FIT, width=2, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=t_smooth, y=forward(t_smooth, T_manual, S_manual), mode="lines",
            name=f"Manual  (T = {T_manual * 86400:.1f} m²/day, S = {S_manual:.2e}; "
                 "other parameters held at their fitted values)",
            line=dict(color=COL_MANUAL, width=2.5),
        ))
        manual_at_obs = forward(t, T_manual, S_manual)
        rmse_mm = float(np.sqrt(np.mean((s_obs - manual_at_obs) ** 2))) * 1000

    fig.update_xaxes(type="log", title_text="Time (s)", showgrid=True, gridcolor=COL_GRID)
    fig.update_yaxes(type="log", title_text="Drawdown (m)", showgrid=True,
                     gridcolor=COL_GRID)
    fig.update_layout(
        title=f"{title or result.method} Manual Matching — RMSE = {rmse_mm:.2f} mm",
        height=480, legend=_LEGEND_H, hovermode="x unified", margin=_MARGIN_SINGLE,
    )
    return fig, rmse_mm


# ── Recovery ──────────────────────────────────────────────────────────────────

def _recovery_line(result, x, T=None, Q=None):
    """The recovery straight line at t/t' = x. With T and Q, the slope is the one
    that T implies and the intercept is held at the fitted S/S'."""
    ratio_0 = result.value("S_ratio", 1.0)
    if T is None:
        slope = result.diagnostics.get("slope_m_per_log_cycle", 0.0)
    else:
        slope = (2.303 * Q) / (4.0 * np.pi * T)
    return slope * (np.log10(x) - np.log10(ratio_0))


def recovery_diagnostic(result: CalculationResult) -> go.Figure:
    """Residual drawdown against t/t' on a semi-log plot — the Theis recovery
    plot. The fitted line should reach s' = 0 near t/t' = 1."""
    from src.calculator.methods.theis_recovery import plot_axis

    ratio, fitted = plot_axis(result)
    s_obs = np.asarray(result.drawdown_obs, dtype=float)
    s_fit = np.asarray(result.drawdown_fitted, dtype=float)
    t_p_h = result.diagnostics.get("pumping_duration_s", 0.0) / 3600.0

    fig = _two_panel(
        f"{result.method} — Residual Drawdown vs t/t′   (t_p = {t_p_h:.2f} h, "
        f"R² = {result.r_squared:.4f}, RMSE = {result.rmse * 1000:.2f} mm)",
        "Residuals (readings used in the fit)")
    fig.add_trace(go.Scatter(
        x=ratio[fitted], y=s_obs[fitted], mode="markers", name="Used in the fit",
        marker=dict(color=COL_OBS, size=7, symbol="circle-open", line=dict(width=1.5)),
        hovertemplate="t/t′ = %{x:.3g}<br>s′ = %{y:.4f} m<extra></extra>",
    ), row=1, col=1)
    if (~fitted).any():
        fig.add_trace(go.Scatter(
            x=ratio[~fitted], y=s_obs[~fitted], mode="markers",
            name="Early recovery (not fitted)",
            marker=dict(color="#888888", size=6, symbol="x"),
            hovertemplate="t/t′ = %{x:.3g}<br>s′ = %{y:.4f} m<extra></extra>",
        ), row=1, col=1)

    x_line = np.logspace(0, np.log10(max(float(ratio.max()), 1.0001)), 120)
    fig.add_trace(go.Scatter(
        x=x_line, y=_recovery_line(result, x_line), mode="lines",
        name=f"Recovery line  ({_parameter_summary(result)})",
        line=dict(color=COL_FIT, width=2),
    ), row=1, col=1)
    # s′ = 0 is where the recovery line must pass at t/t′ = 1.
    fig.add_hline(y=0, line=dict(color=COL_ZERO, width=1, dash="dot"), row=1, col=1)

    _add_residuals(fig, ratio[fitted], s_obs[fitted] - s_fit[fitted], result.rmse)

    _label_panels(fig, "t/t′  (time since pumping started ÷ time since it stopped)",
                  "Residual drawdown s′ (m)", y_log=False)
    fig.update_layout(height=560, legend=_LEGEND_H, margin=_MARGIN_DUAL,
                      hovermode="x unified")
    return fig


def recovery_interactive(result: CalculationResult, Q: float,
                         T_manual: float) -> tuple:
    """Manual matching on the recovery plot. Only T moves — it sets the slope;
    the intercept stays at the fitted S/S'. RMSE is over the fitted readings,
    the same basis as the automatic fit's."""
    from src.calculator.methods.theis_recovery import plot_axis

    ratio, fitted = plot_axis(result)
    s_obs = np.asarray(result.drawdown_obs, dtype=float)
    x_line = np.logspace(0, np.log10(max(float(ratio.max()), 1.0001)), 120)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ratio, y=s_obs, mode="markers", name="Observed",
        marker=dict(color=COL_OBS, size=7, symbol="circle-open", line=dict(width=1.5)),
    ))
    fig.add_trace(go.Scatter(
        x=x_line, y=_recovery_line(result, x_line, result.T, Q), mode="lines",
        name=f"Auto fit  ({_parameter_summary(result)})",
        line=dict(color=COL_FIT, width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=x_line, y=_recovery_line(result, x_line, T_manual, Q), mode="lines",
        name=f"Manual  (T = {T_manual * 86400:.1f} m²/day; S/S′ held at its fit)",
        line=dict(color=COL_MANUAL, width=2.5),
    ))
    manual = _recovery_line(result, ratio[fitted], T_manual, Q)
    rmse_mm = float(np.sqrt(np.mean((s_obs[fitted] - manual) ** 2))) * 1000

    fig.update_xaxes(type="log", title_text="t/t′", showgrid=True, gridcolor=COL_GRID)
    fig.update_yaxes(title_text="Residual drawdown s′ (m)", showgrid=True,
                     gridcolor=COL_GRID)
    fig.update_layout(
        title=f"{result.method} Manual Matching — RMSE = {rmse_mm:.2f} mm",
        height=480, legend=_LEGEND_H, hovermode="x unified", margin=_MARGIN_SINGLE,
    )
    return fig, rmse_mm
