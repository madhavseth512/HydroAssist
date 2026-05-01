"""HydroAssist — Streamlit Web Interface (Phase 1 + Phase 2)

Three UI modes driven by AgentState:
  consultation  → chat with the RAG consultant
  calculation   → calculator workspace (CSV upload → fit → plots)
  Both modes coexist: sidebar always shows chat; main area switches on confirmation.
"""

import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from scipy.special import expn

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from src.core.graph import HydroAssistChat
from src.core.config import load_config
from langchain_core.messages import HumanMessage, AIMessage

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HydroAssist",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header { font-size: 2.2rem; color: #1E88E5; font-weight: bold; }
.subtitle    { font-size: 0.95rem; color: #666; margin-bottom: 1.5rem; }
.result-box  { padding: 1rem; background: #f8f9fa; border-radius: 8px;
               border-left: 4px solid #1E88E5; margin-bottom: 0.8rem;
               color: #212121 !important; }
.warn-box    { padding: 0.8rem; background: #fff8e1; border-radius: 6px;
               border-left: 4px solid #f9a825; margin-bottom: 0.6rem;
               color: #212121 !important; }
</style>
""", unsafe_allow_html=True)


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource
def load_chat_system(_hash=None):
    try:
        config = load_config()
        return HydroAssistChat(config), None
    except Exception as e:
        return None, str(e)


def _config_hash():
    p = Path("configs/default.yaml")
    import hashlib
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else "default"


# ── Session state init ────────────────────────────────────────────────────────
if "chat" not in st.session_state:
    chat, err = load_chat_system(_hash=_config_hash())
    if err:
        st.error(f"Failed to initialise HydroAssist: {err}")
        st.stop()
    st.session_state.chat = chat

if "calc_result" not in st.session_state:
    st.session_state.calc_result = None   # last CalculationResult
if "calc_inputs" not in st.session_state:
    st.session_state.calc_inputs = {}     # Q, r, method, n_excluded
if "formatted_df" not in st.session_state:
    st.session_state.formatted_df = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def _run_calculator(df, Q_si, r, method, r_w=0.10, r_c=0.10):
    """Run the selected calculator and return CalculationResult."""
    from src.calculator.base import CalculationInput
    from src.calculator.theis import TheisCalculator
    from src.calculator.cooper_jacob import CooperJacobCalculator
    from src.calculator.papadopulos_cooper import PapadopulosCooperCalculator

    if "Theis" in method:
        inp = CalculationInput(df=df, Q=Q_si, r=r)
        return TheisCalculator().calculate(inp)
    elif "Cooper-Jacob" in method:
        inp = CalculationInput(df=df, Q=Q_si, r=r)
        return CooperJacobCalculator().calculate(inp)
    else:
        # Papadopulos-Cooper: data from pumped well, r = r_w
        inp = CalculationInput(df=df, Q=Q_si, r=r_w, r_w=r_w, r_c=r_c)
        return PapadopulosCooperCalculator().calculate(inp)


def _format_ci(ci, scale=86400, unit="m²/day"):
    if ci is None:
        return "not available"
    return f"[{ci[0]*scale:.2f}, {ci[1]*scale:.2f}] {unit}"


# ── Static plot helpers (must be defined before tab layout runs) ──────────────

def _theis_static_plot(result, Q, r):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    COL_OBS, COL_FIT, COL_GRID = "#1f77b4", "#d62728", "#e0e0e0"

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.75, 0.25], shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Theis (1935) — Log-Log Type Curve", "Residuals"),
    )

    t = result.time_s
    s_obs = result.drawdown_obs

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
        name=f"Theis fit  T={result.T_day:.1f} m²/day  S={result.S:.2e}",
        line=dict(color=COL_FIT, width=2),
    ), row=1, col=1)

    if result.drawdown_fitted is not None:
        res = s_obs - result.drawdown_fitted
        fig.add_trace(go.Bar(
            x=t, y=res*1000, name="Residual (mm)",
            marker_color=[COL_FIT if v < 0 else COL_OBS for v in res], showlegend=False,
        ), row=2, col=1)
        fig.add_hline(y=0, line=dict(color="black", width=0.8), row=2, col=1)

    fig.update_xaxes(type="log", title_text="Time (s)", showgrid=True, gridcolor=COL_GRID)
    fig.update_yaxes(type="log", title_text="Drawdown (m)", showgrid=True, gridcolor=COL_GRID, row=1, col=1)
    fig.update_yaxes(title_text="Residual (mm)", showgrid=True, gridcolor=COL_GRID, row=2, col=1)

    ci_t = f"  CI: [{result.T_ci[0]*86400:.1f}, {result.T_ci[1]*86400:.1f}]" if result.T_ci else ""
    fig.add_annotation(xref="paper", yref="paper", x=0.99, y=0.97, xanchor="right", yanchor="top",
        text=(f"<b>T</b> = {result.T_day:.2f} m²/day{ci_t}<br>"
              f"<b>S</b> = {result.S:.2e}<br>"
              f"<b>R²</b> = {result.r_squared:.4f}<br>"
              f"<b>RMSE</b> = {result.rmse*1000:.2f} mm"),
        showarrow=False, bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc",
        borderwidth=1, font=dict(size=11), row=1, col=1)

    fig.update_layout(height=550, hovermode="x unified",
                      legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
                      margin=dict(t=60, b=40))
    return fig


def _cj_static_plot(result, Q, r, n_excluded):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    COL_OBS, COL_FIT, COL_EXCL, COL_GRID = "#1f77b4", "#d62728", "#aec7e8", "#e0e0e0"

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.75, 0.25], shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Cooper-Jacob (1946) — Semi-Log Straight Line", "Residuals"),
    )

    t = result.time_s
    s_obs = result.drawdown_obs
    s_fit = result.drawdown_fitted
    t_excl, s_excl = t[:n_excluded], s_obs[:n_excluded]
    t_late, s_late = t[n_excluded:], s_obs[n_excluded:]

    if n_excluded > 0:
        fig.add_trace(go.Scatter(
            x=t_excl, y=s_excl, mode="markers",
            name=f"Excluded early-time ({n_excluded} pts)",
            marker=dict(color=COL_EXCL, size=7, symbol="circle-open", line=dict(width=1.5)),
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t_late, y=s_late, mode="markers", name="Observed (u < 0.02)",
        marker=dict(color=COL_OBS, size=7, symbol="circle-open", line=dict(width=1.5)),
    ), row=1, col=1)

    if s_fit is not None:
        fig.add_trace(go.Scatter(
            x=t, y=s_fit, mode="lines",
            name=f"CJ fit  T={result.T_day:.1f} m²/day  S={result.S:.2e}",
            line=dict(color=COL_FIT, width=2),
        ), row=1, col=1)

    t0 = (result.S * r**2) / (2.25 * result.T)
    if t0 > 0:
        fig.add_annotation(x=np.log10(t0), y=0, xref="x", yref="y",
                           text=f"t₀={t0:.0f}s", showarrow=True, arrowhead=2,
                           ax=30, ay=-30, row=1, col=1)

    if s_fit is not None:
        res = s_obs[n_excluded:] - s_fit[n_excluded:]
        fig.add_trace(go.Bar(
            x=t_late, y=res*1000,
            marker_color=[COL_FIT if v < 0 else COL_OBS for v in res], showlegend=False,
        ), row=2, col=1)
        fig.add_hline(y=0, line=dict(color="black", width=0.8), row=2, col=1)

    fig.update_xaxes(type="log", title_text="Time (s)", showgrid=True, gridcolor=COL_GRID)
    fig.update_yaxes(title_text="Drawdown (m)", showgrid=True, gridcolor=COL_GRID, row=1, col=1)
    fig.update_yaxes(title_text="Residual (mm)", showgrid=True, gridcolor=COL_GRID, row=2, col=1)

    ci_t = f"  CI: [{result.T_ci[0]*86400:.1f}, {result.T_ci[1]*86400:.1f}]" if result.T_ci else ""
    fig.add_annotation(xref="paper", yref="paper", x=0.99, y=0.97, xanchor="right", yanchor="top",
        text=(f"<b>T</b> = {result.T_day:.2f} m²/day{ci_t}<br>"
              f"<b>S</b> = {result.S:.2e}<br>"
              f"<b>R²</b> = {result.r_squared:.4f}<br>"
              f"<b>RMSE</b> = {result.rmse*1000:.2f} mm"),
        showarrow=False, bgcolor="rgba(255,255,255,0.85)", bordercolor="#ccc",
        borderwidth=1, font=dict(size=11), row=1, col=1)

    fig.update_layout(height=550, hovermode="x unified",
                      legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
                      margin=dict(t=60, b=40))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — always visible
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="main-header">💧 HydroAssist</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Hydrogeological Consultant + Calculator</p>',
                unsafe_allow_html=True)

    context = st.session_state.chat.get_context()

    c1, c2 = st.columns(2)
    c1.metric("Aquifer", context["aquifer"].capitalize())
    c2.metric("Messages", context["message_count"])

    st.markdown("**Intent:**")
    st.info(context["intent"].capitalize())
    st.markdown("**Method:**")
    st.info(context["method"] or "None detected")

    st.divider()
    st.subheader("⚙️ Controls")

    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.chat.reset()
        st.session_state.calc_result = None
        st.session_state.formatted_df = None
        st.rerun()

    messages = st.session_state.chat.state.get("messages", [])
    if messages and st.button("💾 Export Chat", use_container_width=True):
        txt = "\n\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in messages
        ])
        st.download_button("Download", txt, "hydroassist_chat.txt", "text/plain",
                           use_container_width=True)

    st.divider()
    st.caption("BTP-2 · IIT Kharagpur · Gemini 2.5 Flash")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — two tabs
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="main-header">💧 HydroAssist</p>', unsafe_allow_html=True)

tab_chat, tab_calc = st.tabs(["💬 Consultation", "📐 Calculator Workspace"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Consultation chat (Phase 1)
# ─────────────────────────────────────────────────────────────────────────────
with tab_chat:
    messages = st.session_state.chat.state.get("messages", [])

    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    retrieved_docs = st.session_state.chat.state.get("retrieved_docs", [])
    if retrieved_docs:
        with st.expander(f"📚 Retrieved Sources ({len(retrieved_docs)})"):
            for i, doc in enumerate(retrieved_docs, 1):
                meta = doc.get("metadata", {})
                st.markdown(f"**{i}.** {meta.get('source','Unknown')}  "
                            f"| Method: {meta.get('method','N/A')}  "
                            f"| Aquifer: {meta.get('aquifer','N/A')}  "
                            f"| Score: {doc.get('score',0):.2f}")
                preview = doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
                st.caption(preview)
                st.divider()

    if len(messages) == 0:
        with st.chat_message("assistant"):
            st.markdown("""
**Welcome to HydroAssist!**

I can help with:
- Understanding pumping test methods (Theis, Cooper-Jacob, Papadopulos-Cooper…)
- Selecting the right method for your aquifer conditions
- Explaining assumptions, limitations, and theory

Switch to the **Calculator Workspace** tab to upload field data and run a curve-fitting analysis.

*Try asking:* "When should I use Cooper-Jacob instead of Theis?"
            """)

    if prompt := st.chat_input("Ask about pumping test methods…"):
        with st.chat_message("user"):
            st.write(prompt)
        with st.spinner("Thinking…"):
            try:
                response = st.session_state.chat.send_message(prompt)
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Calculator Workspace (Phase 2)
# ─────────────────────────────────────────────────────────────────────────────
with tab_calc:

    # ── Step 1: CSV upload ────────────────────────────────────────────────────
    st.subheader("Step 1 — Upload Field Data")

    uploaded = st.file_uploader(
        "Upload a CSV file with time and drawdown columns",
        type=["csv"],
        help="Expected columns: time (s/min/hr) and drawdown (m/ft). Multi-well wide format supported.",
    )

    if uploaded:
        try:
            from src.data.csv_inspector import CSVInspector
            from src.data.formatter import DataFormatter
            from src.data.validator import DataValidator

            raw_df = pd.read_csv(uploaded)
            inspector = CSVInspector()
            inspection = inspector.inspect(raw_df)

            st.success(f"Detected: time col = **{inspection.time_col}** "
                       f"({inspection.time_unit}), "
                       f"drawdown col(s) = **{', '.join(inspection.drawdown_cols)}** "
                       f"({inspection.drawdown_unit})  "
                       f"| Confidence: {inspection.confidence:.0%}")

            if inspection.issues:
                for iss in inspection.issues:
                    st.warning(iss)

            formatter = DataFormatter()
            fmt = formatter.format(raw_df, inspection)
            df = fmt.df
            st.session_state.formatted_df = df

            if fmt.conversions_applied:
                with st.expander("Conversions applied"):
                    for c in fmt.conversions_applied:
                        st.caption(f"• {c}")
            if fmt.rows_removed:
                with st.expander("Rows removed"):
                    for r in fmt.rows_removed:
                        st.caption(f"• {r}")

            st.markdown(f"**{len(df)} data points** ready  "
                        f"| t: {df.time_s.min():.0f}s – {df.time_s.max():.0f}s  "
                        f"| s: {df.drawdown_m.min():.3f}m – {df.drawdown_m.max():.3f}m")

            with st.expander("Preview formatted data (first 10 rows)"):
                st.dataframe(df.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"Failed to process CSV: {e}")
            st.session_state.formatted_df = None

    # ── Step 2: Well parameters ───────────────────────────────────────────────
    st.divider()
    st.subheader("Step 2 — Well Parameters")

    col1, col2, col3 = st.columns(3)
    with col1:
        Q_input = st.number_input(
            "Pumping rate Q",
            min_value=0.0001, max_value=100000.0,
            value=500.0, step=10.0,
            help="Enter Q in the unit selected below",
        )
        Q_unit = st.selectbox("Q unit", ["m³/day", "m³/hour", "m³/s", "L/s"])
    with col2:
        r = st.number_input(
            "Observation well distance r (m)",
            min_value=0.1, max_value=5000.0,
            value=50.0, step=1.0,
            help="Distance from pumping well to the observation well",
        )
    with col3:
        method = st.selectbox(
            "Analysis method",
            ["Theis (1935)", "Cooper-Jacob (1946)", "Papadopulos-Cooper (1967)"],
            help=(
                "Theis: full-curve fit, all times. "
                "Cooper-Jacob: late-time semi-log line (u < 0.02). "
                "Papadopulos-Cooper: large-diameter well with wellbore storage."
            ),
        )

    # Papadopulos-Cooper — extra geometry inputs (shown only when PC selected)
    r_w, r_c = 0.10, 0.10
    if "Papadopulos" in method:
        st.info(
            "**Papadopulos-Cooper (1967):** data must be from the **pumped well** itself "
            "(not an observation well). Enter the physical well dimensions below."
        )
        pc_col1, pc_col2 = st.columns(2)
        with pc_col1:
            r_w = st.number_input(
                "Well screen radius r_w (m)",
                min_value=0.01, max_value=5.0, value=0.15, step=0.01,
                help="Effective radius of the well screen or open hole (paper: r_w)",
            )
        with pc_col2:
            r_c = st.number_input(
                "Casing radius r_c (m)",
                min_value=0.01, max_value=5.0, value=0.10, step=0.01,
                help="Radius of the well casing where the water level declines (paper: r_c)",
            )
        st.caption(
            f"α = r_w²·S/r_c² — dimensionless storage parameter. "
            f"r_w = {r_w} m, r_c = {r_c} m"
        )

    # Convert Q to m³/s
    Q_convert = {"m³/day": 1/86400, "m³/hour": 1/3600, "m³/s": 1.0, "L/s": 1e-3}
    Q_si = Q_input * Q_convert[Q_unit]
    st.caption(f"Q = {Q_si:.6f} m³/s")

    # ── Step 3: Run analysis ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Step 3 — Run Analysis")

    run_disabled = st.session_state.formatted_df is None
    if run_disabled:
        st.info("Upload a CSV file in Step 1 to enable analysis.")

    if "Papadopulos" in method:
        st.caption("Note: Papadopulos-Cooper integrates the well function numerically — "
                   "fitting may take 10–30 seconds depending on dataset size.")

    if st.button("▶  Run Analysis", type="primary", disabled=run_disabled, use_container_width=True):
        df = st.session_state.formatted_df
        with st.spinner(f"Running {method}…"):
            try:
                result = _run_calculator(df, Q_si, r, method, r_w=r_w, r_c=r_c)
                st.session_state.calc_result = result
                # count excluded points (CJ only)
                n_excl = 0
                if "Cooper-Jacob" in method and result.success:
                    u_all = (r**2 * result.S) / (4.0 * result.T * df.time_s.values)
                    n_excl = int((u_all >= 0.02).sum())
                st.session_state.calc_inputs = {
                    "Q": Q_si, "r": r, "method": method, "n_excluded": n_excl,
                    "r_w": r_w, "r_c": r_c,
                }
                if result.success:
                    st.success("Analysis complete.")
                else:
                    st.error(f"Analysis failed: {result.error_message}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    # ── Step 4: Results ───────────────────────────────────────────────────────
    result = st.session_state.calc_result
    if result and result.success:
        st.divider()
        st.subheader("Results")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("T (m²/day)", f"{result.T_day:.2f}")
        m2.metric("T (m²/s)", f"{result.T:.3e}")
        m3.metric("S", f"{result.S:.2e}")
        m4.metric("R²", f"{result.r_squared:.4f}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="result-box">'
                        f'<b>T 95% CI:</b> {_format_ci(result.T_ci)}<br>'
                        f'<b>S 95% CI:</b> {_format_ci(result.S_ci, scale=1, unit="")}<br>'
                        f'<b>RMSE:</b> {result.rmse*1000:.2f} mm'
                        '</div>', unsafe_allow_html=True)
        with col_b:
            if result.validity_notes:
                with st.expander(f"Validity notes ({len(result.validity_notes)})"):
                    for note in result.validity_notes:
                        st.markdown(f"• {note}")

        inputs = st.session_state.calc_inputs
        Q_si_r  = inputs.get("Q", Q_si)
        r_r     = inputs.get("r", r)
        method_r = inputs.get("method", method)
        n_excl  = inputs.get("n_excluded", 0)
        r_w_r   = inputs.get("r_w", 0.10)
        r_c_r   = inputs.get("r_c", 0.10)

        # ── Step 4b: PDF Report download ──────────────────────────────────────
        st.divider()
        st.subheader("Step 4b — Download Report")
        if st.button("📄 Generate PDF Report", use_container_width=True):
            from src.calculator.report import generate_pdf
            with st.spinner("Generating PDF…"):
                try:
                    pdf_bytes = generate_pdf(
                        result,
                        Q_si_r, r_r, method_r,
                        st.session_state.formatted_df,
                        n_excl,
                    )
                    fname = (
                        method_r.split("(")[0].strip().lower().replace(" ", "_")
                        + "_report.pdf"
                    )
                    st.download_button(
                        label="⬇ Download PDF Report",
                        data=pdf_bytes,
                        file_name=fname,
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Failed to generate PDF: {e}")

        # ── Step 5: Diagnostic plots ──────────────────────────────────────────
        st.divider()
        st.subheader("Step 5 — Diagnostic Plots")
        plot_tab1, plot_tab2 = st.tabs(["📊 Fit Diagnostic", "🎛️ Manual Matching"])

        with plot_tab1:
            if "Theis" in method_r:
                fig = _theis_static_plot(result, Q_si_r, r_r)
            elif "Cooper-Jacob" in method_r:
                fig = _cj_static_plot(result, Q_si_r, r_r, n_excl)
            else:
                from src.calculator.plots import pc_diagnostic
                fig = pc_diagnostic(result, Q_si_r, r_w_r, r_c_r)
            st.plotly_chart(fig, use_container_width=True)

        with plot_tab2:
            st.markdown("**Adjust T and S with the sliders — the curve updates in real time.**")
            st.markdown("Match the orange curve to the observed data, then compare with the automated fit (red dashed).")

            from src.calculator.plots import theis_interactive, cj_interactive

            log_T_min, log_T_max = -7.0, 0.0   # 1e-7 to 1.0 m²/s
            log_S_min, log_S_max = -7.0, -2.0  # 1e-7 to 1e-2

            sc1, sc2 = st.columns(2)
            with sc1:
                log_T_manual = st.slider(
                    "log₁₀(T)  [T in m²/s]",
                    min_value=log_T_min, max_value=log_T_max,
                    value=float(np.log10(result.T)),
                    step=0.05,
                    format="%.2f",
                    help="Slide to adjust transmissivity",
                )
                T_manual = 10.0 ** log_T_manual
                st.caption(f"T = {T_manual:.3e} m²/s = {T_manual*86400:.2f} m²/day")

            with sc2:
                log_S_manual = st.slider(
                    "log₁₀(S)  [dimensionless]",
                    min_value=log_S_min, max_value=log_S_max,
                    value=float(np.log10(max(result.S, 1e-7))),
                    step=0.05,
                    format="%.2f",
                    help="Slide to adjust storativity",
                )
                S_manual = 10.0 ** log_S_manual
                st.caption(f"S = {S_manual:.2e}")

            if "Theis" in method_r:
                fig_m, rmse_m = theis_interactive(
                    result.time_s, result.drawdown_obs,
                    Q_si_r, r_r,
                    result.T, result.S,
                    T_manual, S_manual,
                )
            elif "Cooper-Jacob" in method_r:
                fig_m, rmse_m = cj_interactive(
                    result.time_s, result.drawdown_obs,
                    Q_si_r, r_r,
                    result.T, result.S,
                    T_manual, S_manual,
                )
            else:
                from src.calculator.plots import pc_interactive
                st.caption("PC manual matching computes the full well function — "
                           "allow 3–5 seconds after each slider move.")
                fig_m, rmse_m = pc_interactive(
                    result.time_s, result.drawdown_obs,
                    Q_si_r, r_w_r, r_c_r,
                    result.T, result.S,
                    T_manual, S_manual,
                )

            st.plotly_chart(fig_m, use_container_width=True)

            col_use1, col_use2, _ = st.columns([1, 1, 3])
            with col_use1:
                st.metric("Manual RMSE", f"{rmse_m:.2f} mm")
            with col_use2:
                st.metric("Auto RMSE", f"{result.rmse*1000:.2f} mm")

            if st.button("Accept manual values as final estimate"):
                st.success(
                    f"Manual estimate accepted:  "
                    f"T = {T_manual:.3e} m²/s ({T_manual*86400:.2f} m²/day),  "
                    f"S = {S_manual:.2e}"
                )

