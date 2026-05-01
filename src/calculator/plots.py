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
COL_OBS     = "#1f77b4"   # blue   — observed data
COL_FIT     = "#d62728"   # red    — fitted / theoretical curve
COL_EXCL    = "#aec7e8"   # light blue — excluded early-time points (CJ)
COL_MANUAL  = "#ff7f0e"   # orange — manual slider curve
COL_GRID    = "#e0e0e0"


# ── Theis static plot ─────────────────────────────────────────────────────────

def theis_diagnostic(result: CalculationResult) -> go.Figure:
    """
    Log-log type curve for Theis (1935).

    Shows observed data as scatter + fitted Theis curve as a smooth line.
    A good fit is visually apparent when the points fall on the curve.
    Residuals are shown as a small subplot below.
    """
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Theis (1935) — Log-Log Type Curve", "Residuals"),
    )

    t = result.time_s
    s_obs = result.drawdown_obs
    s_fit = result.drawdown_fitted

    # Observed data
    fig.add_trace(go.Scatter(
        x=t, y=s_obs,
        mode="markers",
        name="Observed",
        marker=dict(color=COL_OBS, size=7, symbol="circle-open", line=dict(width=1.5)),
        hovertemplate="t = %{x:.1f} s<br>s = %{y:.4f} m<extra></extra>",
    ), row=1, col=1)

    # Fitted Theis curve (smooth)
    t_smooth = np.logspace(np.log10(t.min()), np.log10(t.max()), 300)
    Q, r = _back_calc_Qr(result)
    if Q and r:
        u_smooth = (r**2 * result.S) / (4.0 * result.T * t_smooth)
        s_smooth = (Q / (4.0 * np.pi * result.T)) * expn(1, np.maximum(u_smooth, 1e-10))
        fig.add_trace(go.Scatter(
            x=t_smooth, y=s_smooth,
            mode="lines",
            name=f"Theis fit  T={result.T*86400:.1f} m²/day  S={result.S:.2e}",
            line=dict(color=COL_FIT, width=2),
            hovertemplate="t = %{x:.1f} s<br>s_fit = %{y:.4f} m<extra></extra>",
        ), row=1, col=1)

    # Residuals
    if s_fit is not None:
        residuals = s_obs - s_fit
        fig.add_trace(go.Bar(
            x=t, y=residuals * 1000,
            name="Residual (mm)",
            marker_color=[COL_FIT if r < 0 else COL_OBS for r in residuals],
            hovertemplate="t = %{x:.1f} s<br>res = %{y:.2f} mm<extra></extra>",
            showlegend=False,
        ), row=2, col=1)
        fig.add_hline(y=0, line=dict(color="black", width=0.8), row=2, col=1)

    _apply_theis_axes(fig)
    _add_fit_annotation(fig, result)
    return fig


def _apply_theis_axes(fig: go.Figure):
    fig.update_xaxes(type="log", title_text="Time (s)", showgrid=True,
                     gridcolor=COL_GRID, row=1, col=1)
    fig.update_yaxes(type="log", title_text="Drawdown (m)", showgrid=True,
                     gridcolor=COL_GRID, row=1, col=1)
    fig.update_xaxes(type="log", title_text="Time (s)", showgrid=True,
                     gridcolor=COL_GRID, row=2, col=1)
    fig.update_yaxes(title_text="Residual (mm)", showgrid=True,
                     gridcolor=COL_GRID, row=2, col=1)
    fig.update_layout(
        height=550, legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=60, b=40), hovermode="x unified",
    )


# ── Cooper-Jacob static plot ──────────────────────────────────────────────────

def cj_diagnostic(result: CalculationResult, n_excluded: int = 0) -> go.Figure:
    """
    Semi-log straight-line plot for Cooper-Jacob (1946).

    Late-time valid points in blue, excluded early-time points in grey,
    fitted straight line in red, t0 intercept marked.
    """
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Cooper-Jacob (1946) — Semi-Log Straight Line", "Residuals"),
    )

    t = result.time_s
    s_obs = result.drawdown_obs
    s_fit = result.drawdown_fitted

    # Split into excluded (early) and fitted (late) points
    n_late = len(t) - n_excluded
    t_excl, s_excl = t[:n_excluded], s_obs[:n_excluded]
    t_late, s_late = t[n_excluded:], s_obs[n_excluded:]

    if n_excluded > 0:
        fig.add_trace(go.Scatter(
            x=t_excl, y=s_excl,
            mode="markers",
            name=f"Excluded early-time ({n_excluded} pts, u ≥ 0.02)",
            marker=dict(color=COL_EXCL, size=7, symbol="circle-open", line=dict(width=1.5)),
            hovertemplate="t = %{x:.1f} s<br>s = %{y:.4f} m<extra></extra>",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t_late, y=s_late,
        mode="markers",
        name="Observed (valid, u < 0.02)",
        marker=dict(color=COL_OBS, size=7, symbol="circle-open", line=dict(width=1.5)),
        hovertemplate="t = %{x:.1f} s<br>s = %{y:.4f} m<extra></extra>",
    ), row=1, col=1)

    # Fitted straight line across full time range
    if s_fit is not None:
        fig.add_trace(go.Scatter(
            x=t, y=s_fit,
            mode="lines",
            name=f"CJ fit  T={result.T*86400:.1f} m²/day  S={result.S:.2e}",
            line=dict(color=COL_FIT, width=2),
            hovertemplate="t = %{x:.1f} s<br>s_fit = %{y:.4f} m<extra></extra>",
        ), row=1, col=1)

    # t0 marker (zero-drawdown intercept)
    t0 = _compute_t0(result)
    if t0 and t0 > 0:
        fig.add_vline(
            x=np.log10(t0),   # plotly log axis needs the log10 value for vline
            line=dict(color="grey", dash="dash", width=1),
            annotation_text=f"t₀ = {t0:.1f} s",
            annotation_position="top right",
            row=1, col=1,
        )

    # Residuals
    if s_fit is not None:
        residuals = s_obs[n_excluded:] - s_fit[n_excluded:]
        fig.add_trace(go.Bar(
            x=t_late, y=residuals * 1000,
            name="Residual (mm)",
            marker_color=[COL_FIT if r < 0 else COL_OBS for r in residuals],
            showlegend=False,
            hovertemplate="t = %{x:.1f} s<br>res = %{y:.2f} mm<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(y=0, line=dict(color="black", width=0.8), row=2, col=1)

    fig.update_xaxes(type="log", title_text="Time (s)", showgrid=True,
                     gridcolor=COL_GRID, row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (m)", showgrid=True,
                     gridcolor=COL_GRID, row=1, col=1)
    fig.update_xaxes(type="log", title_text="Time (s)", showgrid=True,
                     gridcolor=COL_GRID, row=2, col=1)
    fig.update_yaxes(title_text="Residual (mm)", showgrid=True,
                     gridcolor=COL_GRID, row=2, col=1)
    fig.update_layout(
        height=550, legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=60, b=40), hovermode="x unified",
    )
    _add_fit_annotation(fig, result)
    return fig


# ── Theis interactive (manual matching) ───────────────────────────────────────

def theis_interactive(
    time_s: np.ndarray,
    drawdown_obs: np.ndarray,
    Q: float,
    r: float,
    T_fit: float,
    S_fit: float,
    T_manual: float,
    S_manual: float,
) -> go.Figure:
    """
    Log-log plot with the automated fit overlaid alongside the manual curve.
    T_manual, S_manual come from Streamlit sliders in the UI layer.
    """
    fig = go.Figure()

    # Observed
    fig.add_trace(go.Scatter(
        x=time_s, y=drawdown_obs,
        mode="markers",
        name="Observed data",
        marker=dict(color=COL_OBS, size=7, symbol="circle-open", line=dict(width=1.5)),
        hovertemplate="t = %{x:.1f} s<br>s = %{y:.4f} m<extra></extra>",
    ))

    t_smooth = np.logspace(np.log10(time_s.min()), np.log10(time_s.max()), 300)

    # Automated fit
    u_fit = (r**2 * S_fit) / (4.0 * T_fit * t_smooth)
    s_auto = (Q / (4.0 * np.pi * T_fit)) * expn(1, np.maximum(u_fit, 1e-10))
    fig.add_trace(go.Scatter(
        x=t_smooth, y=s_auto,
        mode="lines",
        name=f"Auto fit  T={T_fit*86400:.1f} m²/day  S={S_fit:.2e}",
        line=dict(color=COL_FIT, width=2, dash="dot"),
    ))

    # Manual curve
    u_man = (r**2 * S_manual) / (4.0 * T_manual * t_smooth)
    s_man = (Q / (4.0 * np.pi * T_manual)) * expn(1, np.maximum(u_man, 1e-10))
    fig.add_trace(go.Scatter(
        x=t_smooth, y=s_man,
        mode="lines",
        name=f"Manual  T={T_manual*86400:.1f} m²/day  S={S_manual:.2e}",
        line=dict(color=COL_MANUAL, width=2.5),
    ))

    # RMSE of manual curve on observed points
    u_obs = (r**2 * S_manual) / (4.0 * T_manual * time_s)
    s_man_obs = (Q / (4.0 * np.pi * T_manual)) * expn(1, np.maximum(u_obs, 1e-10))
    rmse_mm = float(np.sqrt(np.mean((drawdown_obs - s_man_obs)**2))) * 1000

    fig.update_xaxes(type="log", title_text="Time (s)", showgrid=True, gridcolor=COL_GRID)
    fig.update_yaxes(type="log", title_text="Drawdown (m)", showgrid=True, gridcolor=COL_GRID)
    fig.update_layout(
        title=f"Theis Manual Matching — RMSE = {rmse_mm:.2f} mm",
        height=480,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        hovermode="x unified",
        margin=dict(t=60, b=40),
    )
    return fig, rmse_mm


# ── Cooper-Jacob interactive (manual matching) ────────────────────────────────

def cj_interactive(
    time_s: np.ndarray,
    drawdown_obs: np.ndarray,
    Q: float,
    r: float,
    T_fit: float,
    S_fit: float,
    T_manual: float,
    S_manual: float,
) -> go.Figure:
    """
    Semi-log plot with the automated fit and the manually adjusted straight line.
    T_manual and S_manual come from Streamlit sliders.
    The manual line is back-computed: slope = 2.303Q/(4piT), intercept from S.
    """
    fig = go.Figure()

    # Observed
    fig.add_trace(go.Scatter(
        x=time_s, y=drawdown_obs,
        mode="markers",
        name="Observed data",
        marker=dict(color=COL_OBS, size=7, symbol="circle-open", line=dict(width=1.5)),
        hovertemplate="t = %{x:.1f} s<br>s = %{y:.4f} m<extra></extra>",
    ))

    t_smooth = np.logspace(np.log10(time_s.min()), np.log10(time_s.max()), 300)

    # Automated fit line
    slope_fit = (2.303 * Q) / (4.0 * np.pi * T_fit)
    t0_fit = (2.25 * T_fit * S_fit) / (r**2)  # rearranged eq.9: t0 = r²S/(2.25T) ... wait: S=2.25Tt0/r² → t0 = Sr²/(2.25T)
    t0_fit = (S_fit * r**2) / (2.25 * T_fit)
    intercept_fit = -slope_fit * np.log10(t0_fit)
    s_auto = slope_fit * np.log10(t_smooth) + intercept_fit
    fig.add_trace(go.Scatter(
        x=t_smooth, y=s_auto,
        mode="lines",
        name=f"Auto fit  T={T_fit*86400:.1f} m²/day  S={S_fit:.2e}",
        line=dict(color=COL_FIT, width=2, dash="dot"),
    ))

    # Manual line
    slope_man = (2.303 * Q) / (4.0 * np.pi * T_manual)
    t0_man = (S_manual * r**2) / (2.25 * T_manual)
    intercept_man = -slope_man * np.log10(t0_man)
    s_man = slope_man * np.log10(t_smooth) + intercept_man
    fig.add_trace(go.Scatter(
        x=t_smooth, y=s_man,
        mode="lines",
        name=f"Manual  T={T_manual*86400:.1f} m²/day  S={S_manual:.2e}",
        line=dict(color=COL_MANUAL, width=2.5),
    ))

    # t0 marker for manual line
    if t0_man > 0:
        fig.add_vline(
            x=np.log10(t0_man),
            line=dict(color=COL_MANUAL, dash="dash", width=1),
            annotation_text=f"t₀ = {t0_man:.1f} s",
            annotation_position="top left",
        )

    # RMSE of manual line on observed data
    s_man_obs = slope_man * np.log10(time_s) + intercept_man
    rmse_mm = float(np.sqrt(np.mean((drawdown_obs - s_man_obs)**2))) * 1000

    fig.update_xaxes(type="log", title_text="Time (s)", showgrid=True, gridcolor=COL_GRID)
    fig.update_yaxes(title_text="Drawdown (m)", showgrid=True, gridcolor=COL_GRID)
    fig.update_layout(
        title=f"Cooper-Jacob Manual Matching — RMSE = {rmse_mm:.2f} mm",
        height=480,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        hovermode="x unified",
        margin=dict(t=60, b=40),
    )
    return fig, rmse_mm


# ── Papadopulos-Cooper static diagnostic plot ─────────────────────────────────

def pc_diagnostic(result: CalculationResult, Q: float,
                  r_w: float, r_c: float) -> go.Figure:
    """
    Log-log type curve for Papadopulos-Cooper (1967).

    Shows observed data, the fitted PC curve, and a family of reference type
    curves for adjacent α values so the analyst can visually verify which
    curve family the data match.  Residuals in lower panel.
    """
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

    t = result.time_s
    s_obs = result.drawdown_obs
    s_fit = result.drawdown_fitted

    # Reference type curves for α ×10 and α /10
    t_smooth = np.logspace(np.log10(t.min()), np.log10(t.max()), 120)
    for alpha_ref, label, dash in [
        (alpha_fit * 10, f"α×10 = {alpha_fit*10:.1e}", "dot"),
        (alpha_fit / 10, f"α/10 = {alpha_fit/10:.1e}", "dot"),
    ]:
        if 1e-8 < alpha_ref < 1.0:
            s_ref = pc_drawdown(t_smooth, result.T, result.S, Q, r_w, r_c)
            # Recompute with varied α by adjusting a surrogate S
            S_ref = alpha_ref * r_c ** 2 / r_w ** 2
            if S_MIN_PLOT < S_ref < S_MAX_PLOT:
                s_ref = pc_drawdown(t_smooth, result.T, S_ref, Q, r_w, r_c)
                fig.add_trace(go.Scatter(
                    x=t_smooth, y=s_ref,
                    mode="lines",
                    name=label,
                    line=dict(color="#999999", width=1.2, dash=dash),
                    hoverinfo="skip",
                ), row=1, col=1)

    # Observed data
    fig.add_trace(go.Scatter(
        x=t, y=s_obs,
        mode="markers",
        name="Observed (pumped well)",
        marker=dict(color=COL_OBS, size=7, symbol="circle-open", line=dict(width=1.5)),
        hovertemplate="t = %{x:.1f} s<br>s = %{y:.4f} m<extra></extra>",
    ), row=1, col=1)

    # Fitted PC curve (smooth)
    s_fitted_smooth = pc_drawdown(t_smooth, result.T, result.S, Q, r_w, r_c)
    fig.add_trace(go.Scatter(
        x=t_smooth, y=s_fitted_smooth,
        mode="lines",
        name=f"PC fit  T={result.T*86400:.1f} m²/day  S={result.S:.2e}  α={alpha_fit:.2e}",
        line=dict(color=COL_FIT, width=2),
    ), row=1, col=1)

    # Residuals
    if s_fit is not None:
        residuals = s_obs - s_fit
        fig.add_trace(go.Bar(
            x=t, y=residuals * 1000,
            name="Residual (mm)",
            marker_color=[COL_FIT if v < 0 else COL_OBS for v in residuals],
            showlegend=False,
            hovertemplate="t = %{x:.1f} s<br>res = %{y:.2f} mm<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(y=0, line=dict(color="black", width=0.8), row=2, col=1)

    # Shade wellbore-storage zone (u_w/α > 1 → early time)
    u_w_arr = (r_w ** 2 * result.S) / (4.0 * result.T * t_smooth)
    ws_mask = u_w_arr / alpha_fit > 1.0
    if ws_mask.any():
        t_ws_end = float(t_smooth[ws_mask][-1]) if ws_mask.any() else t.min()
        fig.add_vrect(
            x0=np.log10(t.min()), x1=np.log10(t_ws_end),
            fillcolor="rgba(255,200,100,0.12)",
            line_width=0,
            annotation_text="Wellbore storage zone",
            annotation_position="top left",
            row=1, col=1,
        )

    fig.update_xaxes(type="log", title_text="Time (s)", showgrid=True, gridcolor=COL_GRID)
    fig.update_yaxes(type="log", title_text="Drawdown (m)", showgrid=True, gridcolor=COL_GRID, row=1, col=1)
    fig.update_xaxes(type="log", showgrid=True, gridcolor=COL_GRID, row=2, col=1)
    fig.update_yaxes(title_text="Residual (mm)", showgrid=True, gridcolor=COL_GRID, row=2, col=1)

    ci_t = ""
    if result.T_ci:
        ci_t = f"  95% CI: [{result.T_ci[0]*86400:.1f}, {result.T_ci[1]*86400:.1f}]"
    fig.add_annotation(
        xref="paper", yref="paper", x=0.99, y=0.97,
        xanchor="right", yanchor="top",
        text=(
            f"<b>T</b> = {result.T*86400:.2f} m²/day{ci_t}<br>"
            f"<b>S</b> = {result.S:.2e} (unreliable — paper p.244)<br>"
            f"<b>α</b> = {alpha_fit:.2e}<br>"
            f"<b>R²</b> = {result.r_squared:.4f}<br>"
            f"<b>RMSE</b> = {result.rmse*1000:.2f} mm"
        ),
        showarrow=False,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#cccccc",
        borderwidth=1,
        font=dict(size=11),
        row=1, col=1,
    )
    fig.update_layout(
        height=560,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=60, b=40),
        hovermode="x unified",
    )
    return fig


# ── Papadopulos-Cooper interactive (manual matching) ──────────────────────────

def pc_interactive(
    time_s: np.ndarray,
    drawdown_obs: np.ndarray,
    Q: float,
    r_w: float,
    r_c: float,
    T_fit: float,
    S_fit: float,
    T_manual: float,
    S_manual: float,
) -> tuple:
    """
    Log-log manual-matching plot for Papadopulos-Cooper.

    The manual curve uses the full PC well function.  The auto-fit curve is
    shown for comparison.  Returns (fig, rmse_mm).

    Note: each curve evaluation integrates F(u_w, α) at every time point.
    The slider resolution is coarser (60 points) to keep interaction snappy.
    """
    fig = go.Figure()

    t_smooth = np.logspace(np.log10(time_s.min()), np.log10(time_s.max()), 60)

    # Observed
    fig.add_trace(go.Scatter(
        x=time_s, y=drawdown_obs,
        mode="markers",
        name="Observed data",
        marker=dict(color=COL_OBS, size=7, symbol="circle-open", line=dict(width=1.5)),
        hovertemplate="t = %{x:.1f} s<br>s = %{y:.4f} m<extra></extra>",
    ))

    # Automated fit
    s_auto = pc_drawdown(t_smooth, T_fit, S_fit, Q, r_w, r_c)
    alpha_fit = r_w ** 2 * S_fit / r_c ** 2
    fig.add_trace(go.Scatter(
        x=t_smooth, y=s_auto,
        mode="lines",
        name=f"Auto fit  T={T_fit*86400:.1f} m²/day  α={alpha_fit:.2e}",
        line=dict(color=COL_FIT, width=2, dash="dot"),
    ))

    # Manual curve
    s_man = pc_drawdown(t_smooth, T_manual, S_manual, Q, r_w, r_c)
    alpha_man = r_w ** 2 * S_manual / r_c ** 2
    fig.add_trace(go.Scatter(
        x=t_smooth, y=s_man,
        mode="lines",
        name=f"Manual  T={T_manual*86400:.1f} m²/day  α={alpha_man:.2e}",
        line=dict(color=COL_MANUAL, width=2.5),
    ))

    # RMSE on observed points
    s_man_obs = pc_drawdown(time_s, T_manual, S_manual, Q, r_w, r_c)
    rmse_mm = float(np.sqrt(np.mean((drawdown_obs - s_man_obs) ** 2))) * 1000

    fig.update_xaxes(type="log", title_text="Time (s)", showgrid=True, gridcolor=COL_GRID)
    fig.update_yaxes(type="log", title_text="Drawdown (m)", showgrid=True, gridcolor=COL_GRID)
    fig.update_layout(
        title=f"PC Manual Matching — RMSE = {rmse_mm:.2f} mm",
        height=480,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        hovermode="x unified",
        margin=dict(t=60, b=40),
    )
    return fig, rmse_mm


S_MIN_PLOT = 1e-10
S_MAX_PLOT = 1e-1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _back_calc_Qr(result: CalculationResult):
    """Estimate Q and r from result metadata if available (not stored yet)."""
    return None, None


def _compute_t0(result: CalculationResult) -> float:
    """Compute t₀ = r²S / (2.25T) from CalculationResult."""
    if result.T > 0 and result.S > 0:
        return None   # r not stored in result yet — computed in UI from inputs
    return None


def _add_fit_annotation(fig: go.Figure, result: CalculationResult):
    """Add a text box with T, S, R², RMSE to the top-right of row 1."""
    ci_t = ""
    if result.T_ci:
        ci_t = f"  95% CI: [{result.T_ci[0]*86400:.1f}, {result.T_ci[1]*86400:.1f}] m²/day"
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.99, y=0.97,
        xanchor="right", yanchor="top",
        text=(
            f"<b>T</b> = {result.T*86400:.2f} m²/day{ci_t}<br>"
            f"<b>S</b> = {result.S:.2e}<br>"
            f"<b>R²</b> = {result.r_squared:.4f}<br>"
            f"<b>RMSE</b> = {result.rmse*1000:.2f} mm"
        ),
        showarrow=False,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#cccccc",
        borderwidth=1,
        font=dict(size=11),
        row=1, col=1,
    )
