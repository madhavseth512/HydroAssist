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


# ── Theis static plot ─────────────────────────────────────────────────────────

def theis_diagnostic(result: CalculationResult) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Theis (1935) — Log-Log Type Curve", "Residuals"),
    )

    t      = result.time_s
    s_obs  = result.drawdown_obs
    s_fit  = result.drawdown_fitted

    fig.add_trace(go.Scatter(
        x=t, y=s_obs, mode="markers", name="Observed",
        marker=dict(color=COL_OBS, size=7, symbol="circle-open", line=dict(width=1.5)),
        hovertemplate="t = %{x:.1f} s<br>s = %{y:.4f} m<extra></extra>",
    ), row=1, col=1)

    t_smooth = np.logspace(np.log10(t.min()), np.log10(t.max()), 300)
    Q, r = _back_calc_Qr(result)
    if Q and r:
        u_s = (r**2 * result.S) / (4.0 * result.T * t_smooth)
        s_s = (Q / (4.0 * np.pi * result.T)) * expn(1, np.maximum(u_s, 1e-10))
        fig.add_trace(go.Scatter(
            x=t_smooth, y=s_s, mode="lines", name="Theis fit",
            line=dict(color=COL_FIT, width=2),
            hovertemplate="t = %{x:.1f} s<br>s_fit = %{y:.4f} m<extra></extra>",
        ), row=1, col=1)

    if s_fit is not None:
        res = s_obs - s_fit
        fig.add_trace(go.Bar(
            x=t, y=res * 1000, name="Residual (mm)",
            marker_color=[COL_FIT if v < 0 else COL_OBS for v in res],
            hovertemplate="t = %{x:.1f} s<br>res = %{y:.2f} mm<extra></extra>",
            showlegend=False,
        ), row=2, col=1)
        fig.add_hline(y=0, line=dict(color="white", width=0.8, dash="dash"), row=2, col=1)

    fig.update_xaxes(type="log", title_text="Time (s)", showgrid=True, gridcolor=COL_GRID, row=1, col=1)
    fig.update_yaxes(type="log", title_text="Drawdown (m)", showgrid=True, gridcolor=COL_GRID, row=1, col=1)
    fig.update_xaxes(type="log", title_text="Time (s)", showgrid=True, gridcolor=COL_GRID, row=2, col=1)
    fig.update_yaxes(title_text="Residual (mm)", showgrid=True, gridcolor=COL_GRID, row=2, col=1)
    fig.update_layout(height=560, legend=_LEGEND_H, margin=_MARGIN_DUAL, hovermode="x unified")
    return fig


# ── Cooper-Jacob static plot ──────────────────────────────────────────────────

def cj_diagnostic(result: CalculationResult, n_excluded: int = 0) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Cooper-Jacob (1946) — Semi-Log Straight Line", "Residuals"),
    )

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
            x=t, y=s_fit, mode="lines", name="CJ fit",
            line=dict(color=COL_FIT, width=2),
            hovertemplate="t = %{x:.1f} s<br>s_fit = %{y:.4f} m<extra></extra>",
        ), row=1, col=1)

    t0 = _compute_t0(result)
    if t0 and t0 > 0:
        fig.add_vline(
            x=np.log10(t0), line=dict(color="grey", dash="dash", width=1),
            annotation_text=f"t₀ = {t0:.1f} s", annotation_position="top right",
            row=1, col=1,
        )

    if s_fit is not None:
        res = s_obs[n_excluded:] - s_fit[n_excluded:]
        fig.add_trace(go.Bar(
            x=t_late, y=res * 1000, name="Residual (mm)",
            marker_color=[COL_FIT if v < 0 else COL_OBS for v in res],
            showlegend=False,
            hovertemplate="t = %{x:.1f} s<br>res = %{y:.2f} mm<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(y=0, line=dict(color="white", width=0.8, dash="dash"), row=2, col=1)

    fig.update_xaxes(type="log", title_text="Time (s)", showgrid=True, gridcolor=COL_GRID, row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (m)", showgrid=True, gridcolor=COL_GRID, row=1, col=1)
    fig.update_xaxes(type="log", title_text="Time (s)", showgrid=True, gridcolor=COL_GRID, row=2, col=1)
    fig.update_yaxes(title_text="Residual (mm)", showgrid=True, gridcolor=COL_GRID, row=2, col=1)
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

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f"Papadopulos-Cooper (1967) — Log-Log Type Curve  (α = {alpha_fit:.2e})",
            "Residuals",
        ),
    )

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
        fig.add_trace(go.Bar(
            x=t, y=res * 1000, name="Residual (mm)",
            marker_color=[COL_FIT if v < 0 else COL_OBS for v in res],
            showlegend=False,
            hovertemplate="t = %{x:.1f} s<br>res = %{y:.2f} mm<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(y=0, line=dict(color="white", width=0.8, dash="dash"), row=2, col=1)

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

    fig.update_xaxes(type="log", title_text="Time (s)", showgrid=True, gridcolor=COL_GRID)
    fig.update_yaxes(type="log", title_text="Drawdown (m)", showgrid=True, gridcolor=COL_GRID, row=1, col=1)
    fig.update_xaxes(type="log", showgrid=True, gridcolor=COL_GRID, row=2, col=1)
    fig.update_yaxes(title_text="Residual (mm)", showgrid=True, gridcolor=COL_GRID, row=2, col=1)
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


S_MIN_PLOT = 1e-10
S_MAX_PLOT = 1e-1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _back_calc_Qr(result: CalculationResult):
    return None, None


def _compute_t0(result: CalculationResult):
    if result.T > 0 and result.S > 0:
        return None
    return None
