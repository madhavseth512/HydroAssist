"""HydroAssist — Streamlit Web Interface (Phase 1 + Phase 2)

Three UI modes driven by AgentState:
  consultation  → chat with the RAG consultant
  calculation   → calculator workspace (CSV upload → fit → plots)
  Both modes coexist: sidebar always shows chat; main area switches on confirmation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from src.core.graph import HydroAssistChat
from src.core.config import load_config
from src.calculator import registry as _registry
from src.project import workspace as _ws
from src.project.model import FILE_SUFFIX, Project, ProjectFileError
from src.services import analysis as _analysis, data_intake as _intake
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


# ── The conversation, one per session ─────────────────────────────────────────
# Deliberately NOT st.cache_resource. HydroAssistChat holds the conversation
# itself — messages, the method suggestion, the recorded calculation — and a
# cached object is shared by every session in the process, so on a hosted
# deployment every visitor would read and write one conversation. The expensive
# parts it builds on are already shared safely at process level: the embedding
# model is a singleton and the vector store is a read-only snapshot.
def load_chat_system():
    try:
        return HydroAssistChat(load_config()), None
    except Exception as e:
        return None, str(e)


# ── Session state init ────────────────────────────────────────────────────────
if "chat" not in st.session_state:
    chat, err = load_chat_system()
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
if "validation" not in st.session_state:
    st.session_state.validation = None   # last ValidationResult
if "diagnosis" not in st.session_state:
    st.session_state.diagnosis = None    # last DerivativeSignature

# The open project, and which of its tests the workspace is showing.
if "project" not in st.session_state:
    st.session_state.project = _ws.new_project()
    st.session_state.active_test_id = st.session_state.project.tests[0].id
# Bumped whenever a different project or test is loaded, so keyed detail
# inputs are recreated showing the new values.
if "project_gen" not in st.session_state:
    st.session_state.project_gen = 0
# Step 2 inputs are keyed so a reopened project can prefill them.
for _key, _default in _ws.DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ── Helpers ───────────────────────────────────────────────────────────────────
# The work itself lives in src/services, which knows nothing about Streamlit;
# this file is the widgets and the session state around it.

_format_ci = _analysis.format_ci

ALL_METHODS = _registry.labels()


def _default_method_index(options):
    """Preselect the method the consultant suggested in chat, if it is offered."""
    return _analysis.default_method_index(
        options, st.session_state.chat.state.get("selected_method", "") or "")


@st.cache_resource
def _inspection_llm():
    """Gemini client for the CSVInspector's layer-3 column-mapping fallback.

    Returns None if the LLM cannot be constructed (no API key, offline) — the
    inspector then degrades to regex + statistical detection rather than
    failing the upload.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        cfg = load_config()
        return ChatGoogleGenerativeAI(
            model=cfg.llm.model,
            temperature=0.0,        # deterministic column mapping
            max_tokens=1024,
            google_api_key=cfg.llm.api_key,
        )
    except Exception:
        return None


# ── Project helpers ───────────────────────────────────────────────────────────

def _active_test():
    """The project test the workspace is showing. Recovers if the stored id no
    longer exists, e.g. just after a different project was opened."""
    project = st.session_state.project
    try:
        return project.get_test(st.session_state.get("active_test_id", ""))
    except KeyError:
        if not project.tests:
            project.add_test()
        st.session_state.active_test_id = project.tests[0].id
        return project.tests[0]


def _load_test_into_workspace(test):
    """Point the workspace at `test`: its readings, its setup in the Step 2 form,
    and a clean slate for results.

    Writes Step 2 widget keys, which Streamlit only allows before those widgets
    render in the run — so this is called only from the sidebar, which executes
    before the workspace tab, and always followed by st.rerun().
    """
    st.session_state.active_test_id = test.id
    st.session_state.calc_result = None
    st.session_state.validation = None
    st.session_state.diagnosis = None
    st.session_state.formatted_df = (
        test.readings.to_frame() if test.readings is not None else None)
    for key, value in _ws.inputs_from_setup(test.setup).items():
        st.session_state[key] = value
    st.session_state.chat.clear_calculation()
    st.session_state.chat.clear_staged_series()
    st.session_state.project_gen = st.session_state.get("project_gen", 0) + 1


def _intake_settings():
    """What the validator needs to judge a dataset, from the current form."""
    inputs = st.session_state.calc_inputs
    return dict(
        test_type=st.session_state.get(_ws.KEY_TEST_TYPE, _registry.CONSTANT_RATE),
        well_radius_m=inputs.get("r"),
        pumping_rate_m3d=inputs.get("Q_day"),
        well_casing_radius_m=inputs.get("r_c"),
    )


def _render_prepared_data(prepared):
    """Everything Step 1 shows for a ready dataset, whatever its source — a fresh
    CSV upload or readings reopened from a project."""
    from src.calculator.plots import derivative_diagnostic

    df = prepared.df
    st.session_state.formatted_df = df
    st.session_state.diagnosis = prepared.diagnosis
    st.session_state.validation = prepared.validation

    # Publish the series to the conversation so the Manager's method suggestion
    # reads the actual flow regime off the derivative rather than inferring it
    # from the words in the user's message.
    st.session_state.chat.stage_series(
        df.time_s.values, df.drawdown_m.values,
        Q=st.session_state.calc_inputs.get("Q"),
    )

    st.markdown(f"**{prepared.summary()}**")

    with st.expander("Preview formatted data (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)

    if prepared.conversions:
        with st.expander("Conversions applied"):
            for c in prepared.conversions:
                st.caption(f"• {c}")
    if prepared.rows_removed:
        with st.expander("Rows removed"):
            for r in prepared.rows_removed:
                st.caption(f"• {r}")

    # Modern practice reads the derivative first and picks the model from its
    # shape, rather than fitting a model chosen in advance and checking R²
    # afterwards. So this sits in Step 1, ahead of the method dropdown.
    with st.expander("🔍 Diagnostic plot — what does the data say?", expanded=True):
        if prepared.diagnosis is None:
            st.caption(f"Diagnostic plot unavailable: {prepared.diagnosis_error}")
        else:
            try:
                st.plotly_chart(
                    derivative_diagnostic(df.time_s.values, df.drawdown_m.values),
                    use_container_width=True,
                )
            except Exception as e:
                st.caption(f"Diagnostic plot unavailable: {e}")
            st.info(f"**{prepared.diagnosis.regime.replace('_', ' ').title()}** "
                    f"— {prepared.diagnosis.evidence}")
            for note in prepared.diagnosis.notes:
                st.caption(f"• {note}")
            if prepared.consistent_methods:
                st.caption("Consistent with: "
                           + ", ".join(prepared.consistent_methods))

    validation = prepared.validation
    for err in validation.errors:
        st.error(err)
    for warn in validation.warnings:
        st.warning(warn)
    if validation.info:
        with st.expander(f"Data notes ({len(validation.info)})"):
            for note in validation.info:
                st.caption(f"• {note}")
    if not prepared.is_usable:
        st.error("This dataset cannot be analysed as-is — resolve the errors "
                 "above before running a fit.")
        st.session_state.formatted_df = None


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
    method_line = context["method"] or "None detected"
    if context["suggestion_confirmed"] and context["method"]:
        method_line += "  ✅ confirmed"
    st.info(method_line)

    if context["has_result"]:
        st.success("Analysis loaded — the consultant can discuss it.")

    # ── Project ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📁 Project")
    project = st.session_state.project
    active = _active_test()
    gen = st.session_state.project_gen
    st.caption(f"**{project.info.title}** · {_ws.project_summary(project)}")

    # Keyed by generation: after a successful open the key changes, so the
    # uploader comes back empty instead of re-opening the same file every rerun.
    opened = st.file_uploader(
        "Open a project", type=["json"], key=f"project_file_{gen}",
        help=f"A {FILE_SUFFIX} file saved from HydroAssist.",
    )
    if opened is not None:
        try:
            loaded = Project.from_json(opened.getvalue())
        except ProjectFileError as e:
            st.error(str(e))
        else:
            if not loaded.tests:
                loaded.add_test()
            st.session_state.project = loaded
            _load_test_into_workspace(loaded.tests[0])
            st.rerun()

    test_ids = [t.id for t in project.tests]
    chosen = st.selectbox(
        "Active test", test_ids, index=test_ids.index(active.id),
        format_func=lambda test_id: project.get_test(test_id).name,
    )
    if chosen != active.id:
        _load_test_into_workspace(project.get_test(chosen))
        st.rerun()

    with st.expander("Project and test details"):
        info = project.info
        info.title = st.text_input("Project title", info.title, key=f"pi_title_{gen}")
        info.client = st.text_input("Client", info.client, key=f"pi_client_{gen}")
        info.project_number = st.text_input(
            "Project number", info.project_number, key=f"pi_number_{gen}")
        info.location = st.text_input("Location", info.location, key=f"pi_loc_{gen}")
        info.test_date = st.text_input("Test date", info.test_date,
                                       placeholder="YYYY-MM-DD", key=f"pi_date_{gen}")
        info.analyst = st.text_input("Analyst", info.analyst, key=f"pi_analyst_{gen}")
        info.notes = st.text_area("Notes", info.notes, key=f"pi_notes_{gen}")
        st.markdown("**Active test**")
        active.name = st.text_input(
            "Test name", active.name, key=f"t_name_{active.id}_{gen}") or active.name
        active.pumping_well = st.text_input(
            "Pumping well", active.pumping_well, key=f"t_pw_{active.id}_{gen}")
        active.observation_well = st.text_input(
            "Observation well", active.observation_well, key=f"t_ow_{active.id}_{gen}")

    pc1, pc2 = st.columns(2)
    if pc1.button("➕ Add test", use_container_width=True):
        _load_test_into_workspace(project.add_test())
        st.rerun()
    pc2.download_button(
        "💾 Save", data=project.to_json(), file_name=project.filename,
        mime="application/json", use_container_width=True,
        help="Download the project as a file you can reopen later.",
    )

    with st.expander("Start a new project"):
        st.caption("Anything not saved to a file will be lost.")
        if st.button("Discard and start new", use_container_width=True):
            st.session_state.project = _ws.new_project()
            _load_test_into_workspace(st.session_state.project.tests[0])
            st.rerun()

    st.divider()
    st.subheader("⚙️ Controls")

    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.chat.reset()
        st.session_state.calc_result = None
        st.session_state.formatted_df = None
        st.session_state.validation = None
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

    # ── Method suggestion card — the chat → calculator bridge ────────────────
    suggestion = st.session_state.chat.state.get("method_suggestion") or {}
    confirmed = st.session_state.chat.state.get("suggestion_confirmed", False)

    if suggestion and not confirmed:
        if not suggestion.get("supported", True) or not suggestion.get("method"):
            st.warning(suggestion.get("reason", ""))
        else:
            with st.container(border=True):
                st.markdown(
                    f"#### 📐 Suggested method: {suggestion['method']}\n\n"
                    f"{suggestion['reason']}"
                )
                st.progress(
                    float(suggestion.get("confidence", 0.0)),
                    text=f"Confidence {suggestion.get('confidence', 0):.0%}",
                )
                b1, b2 = st.columns(2)
                if b1.button("✅ Use this method", use_container_width=True,
                             type="primary"):
                    st.session_state.chat.confirm_method(suggestion["method"])
                    st.rerun()
                override = b2.selectbox(
                    "…or choose another", ALL_METHODS,
                    index=ALL_METHODS.index(suggestion["method"])
                    if suggestion["method"] in ALL_METHODS else 0,
                    label_visibility="collapsed",
                )
                if b2.button("Use selected instead", use_container_width=True):
                    st.session_state.chat.confirm_method(override)
                    st.rerun()

    elif confirmed:
        st.success(
            f"Method set to **{st.session_state.chat.state.get('selected_method')}** "
            "— it is preselected in the Calculator Workspace tab."
        )

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
        # One uploader per test. Otherwise switching tests leaves the previous
        # CSV in the uploader, and the next rerun would silently overwrite the
        # newly selected test's readings with it.
        key=f"csv_{_active_test().id}",
    )

    if uploaded:
        try:
            # A new file invalidates any result already published to the chat,
            # otherwise the consultant would discuss numbers from the old dataset.
            if st.session_state.get("uploaded_name") != uploaded.name:
                st.session_state.uploaded_name = uploaded.name
                st.session_state.calc_result = None
                st.session_state.chat.clear_calculation()
                st.session_state.chat.clear_staged_series()

            # The inspector gets the same Gemini client the agents use, so its
            # layer-3 fallback on ambiguous column names is not dead code.
            prepared = _intake.prepare_upload(
                pd.read_csv(uploaded), llm=_inspection_llm(), **_intake_settings())
            inspection = prepared.inspection

            st.success(f"Detected: time col = **{inspection.time_col}** "
                       f"({inspection.time_unit}), "
                       f"drawdown col(s) = **{', '.join(inspection.drawdown_cols)}** "
                       f"({inspection.drawdown_unit})  "
                       f"| Confidence: {inspection.confidence:.0%}")
            for iss in inspection.issues:
                st.warning(iss)

            # Record the readings in the active project test, so a saved project
            # reopens with its data.
            _intake.store_readings(_active_test(), prepared.df,
                                   source_name=uploaded.name,
                                   conversions=prepared.conversions)

            _render_prepared_data(prepared)

        except Exception as e:
            st.error(f"Failed to process CSV: {e}")
            st.session_state.formatted_df = None
            st.session_state.validation = None

    elif _active_test().readings is not None:
        # No new upload, but the active test already holds readings — typically
        # a reopened project. Use them rather than showing an empty workspace.
        saved = _active_test().readings
        source = f" from {saved.source_name}" if saved.source_name else ""
        st.info(f"Using {len(saved)} readings saved in this project{source}. "
                "Upload a CSV above to replace them.")
        _render_prepared_data(
            _intake.prepare(saved.to_frame(), **_intake_settings()))

    # ── Step 2: Well parameters ───────────────────────────────────────────────
    st.divider()
    st.subheader("Step 2 — Well Parameters")

    # The kind of test decides which methods apply and what else must be entered.
    test_type = st.selectbox(
        "Test type", list(_ws.TEST_TYPE_LABELS), key=_ws.KEY_TEST_TYPE,
        format_func=_ws.TEST_TYPE_LABELS.get,
        help=("Constant-rate: drawdown while the well pumps. Recovery: the water "
              "level rising back after the pump is switched off."),
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        # Keyed (values come from session state) so a reopened project or a
        # switched test can prefill them.
        Q_input = st.number_input(
            "Pumping rate Q",
            min_value=0.0001, max_value=100000.0, step=10.0,
            key=_ws.KEY_Q,
            help="Enter Q in the unit selected below",
        )
        Q_unit = st.selectbox("Q unit", _ws.Q_UNITS, key=_ws.KEY_Q_UNIT)
    with col2:
        r = st.number_input(
            "Observation well distance r (m)",
            min_value=0.1, max_value=5000.0, step=1.0,
            key=_ws.KEY_R,
            help="Distance from pumping well to the observation well",
        )
    with col3:
        method_options, gated_out = _analysis.method_options(
            st.session_state.validation, test_type)
        method = st.selectbox(
            "Analysis method",
            method_options,
            index=_default_method_index(method_options),
            help=_analysis.method_help(method_options),
        )
        if gated_out:
            st.caption(
                "Not offered for this dataset: " + ", ".join(gated_out) +
                " — see the data notes in Step 1."
            )

    # Extra well geometry, shown for methods fitted at the pumped well face.
    # Driven by the registry's data_source rather than by the method's name.
    spec = _registry.require(method)
    needs_well_geometry = spec.data_source == _registry.PUMPED_WELL

    r_w, r_c = 0.10, 0.10
    if needs_well_geometry:
        st.info(
            f"**{spec.label}:** data must be from the **pumped well** itself "
            "(not an observation well). Enter the physical well dimensions below."
        )
        pc_col1, pc_col2 = st.columns(2)
        with pc_col1:
            r_w = st.number_input(
                "Well screen radius r_w (m)",
                min_value=0.01, max_value=5.0, step=0.01, key=_ws.KEY_RW,
                help="Effective radius of the well screen or open hole (paper: r_w)",
            )
        with pc_col2:
            r_c = st.number_input(
                "Casing radius r_c (m)",
                min_value=0.01, max_value=5.0, step=0.01, key=_ws.KEY_RC,
                help="Radius of the well casing where the water level declines (paper: r_c)",
            )
        st.caption(
            f"α = r_w²·S/r_c² — dimensionless storage parameter. "
            f"r_w = {r_w} m, r_c = {r_c} m"
        )

    # Recovery needs to know when the pump stopped, and which clock the time
    # column keeps.
    t_p_s, clock = None, _ws.DEFAULTS[_ws.KEY_CLOCK]
    if test_type == _registry.RECOVERY:
        rc1, rc2 = st.columns(2)
        with rc1:
            t_p_hours = st.number_input(
                "Pumping duration before shut-down t_p (hours)",
                min_value=0.01, max_value=10000.0, step=0.5, key=_ws.KEY_TP,
                help="How long the well pumped at Q before it was switched off.",
            )
            t_p_s = t_p_hours * 3600.0
        with rc2:
            clock = st.radio(
                "The time column counts", list(_ws.CLOCK_LABELS), key=_ws.KEY_CLOCK,
                format_func=_ws.CLOCK_LABELS.get,
                help=("Loggers usually keep one clock for the whole test; manual "
                      "recovery readings usually restart it at shut-down."),
            )

    # Convert Q to m³/s
    Q_si = _ws.q_to_si(Q_input, Q_unit)
    st.caption(f"Q = {Q_si:.6f} m³/s")

    # Keep the active test's setup in step with the form, so saving the
    # project captures what is on screen. The same setup drives the run.
    setup = _ws.setup_from_inputs(st.session_state)
    _active_test().setup = setup

    # ── Step 3: Run analysis ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Step 3 — Run Analysis")

    run_disabled = st.session_state.formatted_df is None
    if run_disabled:
        st.info("Upload a CSV file in Step 1 to enable analysis.")

    if needs_well_geometry:
        st.caption(f"Note: {spec.label} inverts the well function numerically "
                   "(Stehfest Laplace inversion).")

    if st.button("▶  Run Analysis", type="primary", disabled=run_disabled, use_container_width=True):
        df = st.session_state.formatted_df
        with st.spinner(f"Running {method}…"):
            try:
                result = _analysis.run_analysis(df, method, setup)
                st.session_state.calc_result = result
                st.session_state.calc_inputs = _analysis.run_record(
                    setup, method, result, n_points=len(df))

                # ── Publish the result back into the conversation ────────────
                # This is what lets the Consultation tab discuss these numbers.
                st.session_state.chat.record_calculation(
                    result.to_state_dict(),
                    _analysis.chat_record(st.session_state.calc_inputs),
                )
                st.session_state.chat.confirm_method(method)

                if result.success:
                    st.success(
                        "Analysis complete — the Consultation tab can now "
                        "discuss these results."
                    )
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
        if _analysis.estimates_S(result):
            m3.metric("S", f"{result.S:.2e}")
        else:
            m3.metric("S", "not determined",
                      help="This method does not estimate storativity.")
        m4.metric("R²", f"{result.r_squared:.4f}")

        # Methods beyond the original three estimate more than T and S —
        # Hantush-Jacob's leakage factor, Boulton's specific yield, Warren-Root's
        # storativity ratio. The metric row above cannot show those, so list every
        # parameter, flagging the ones that are derived, pinned at a bound, or
        # that the method itself says not to trust.
        if any(p.name not in ("T", "S") for p in result.parameters):
            def _cell(p):
                if p.display_unit and p.display_unit != p.unit:
                    return f"{p.display:.4g}", p.display_unit.replace("^2", "²")
                return f"{p.value:.4g}", ("" if p.unit == "-" else p.unit)

            def _ci(p):
                if p.ci is None:
                    return "derived" if p.derived else "—"
                lo, hi = p.display_ci if p.display_unit != p.unit else p.ci
                return f"{lo:.3g} – {hi:.3g}"

            rows = []
            for p in result.parameters:
                value, unit = _cell(p)
                notes = []
                if p.derived:
                    notes.append("derived")
                if p.at_bound:
                    notes.append(f"at {p.at_bound} bound")
                if not p.reliable:
                    notes.append(p.reliability_note or "low reliability")
                rows.append({"Parameter": p.symbol, "Value": value, "Unit": unit,
                             "95% CI": _ci(p), "Notes": "; ".join(notes)})
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)

        # ── Hand off to the consultant ────────────────────────────────────────
        ask_col, _ = st.columns([2, 3])
        with ask_col:
            if st.button("💬 Interpret these results in chat",
                         use_container_width=True):
                with st.spinner("Asking the consultant…"):
                    try:
                        st.session_state.chat.send_message(
                            "Interpret the pumping test results I just computed. "
                            "Is the fit trustworthy, are T and S physically "
                            "plausible for this aquifer, and what do the "
                            "validity notes mean in practice?"
                        )
                        st.success("Answered in the Consultation tab.")
                    except Exception as e:
                        st.error(f"Consultant unavailable: {e}")

        col_a, col_b = st.columns(2)
        with col_a:
            s_ci_line = (f'<b>S 95% CI:</b> {_format_ci(result.S_ci, scale=1, unit="")}<br>'
                         if _analysis.estimates_S(result) else "")
            st.markdown('<div class="result-box">'
                        f'<b>T 95% CI:</b> {_format_ci(result.T_ci)}<br>'
                        + s_ci_line +
                        f'<b>RMSE:</b> {result.rmse*1000:.2f} mm'
                        '</div>', unsafe_allow_html=True)
        with col_b:
            if result.validity_notes:
                with st.expander(f"Validity notes ({len(result.validity_notes)})"):
                    for note in result.validity_notes:
                        st.markdown(f"• {note}")

        # The setup the fit ran with, which the form may since have moved on from.
        inputs = st.session_state.calc_inputs
        method_r = inputs.get("method", method)

        # ── Save to the project ───────────────────────────────────────────────
        active_test = _active_test()
        # Compared by identity against a held reference, not by id(): an id can
        # be reused once the old result is garbage-collected.
        saved_already = st.session_state.get("saved_result") is result
        if active_test.readings is None:
            st.caption("Load readings into this project test to save analyses.")
        elif saved_already:
            st.success(f"Saved to **{active_test.name}** — compare it with other "
                       "methods in the table at the bottom of this page.")
        elif st.button(f"📌 Save this analysis to “{active_test.name}”",
                       use_container_width=True):
            # Recorded with the setup the fit ran with, not whatever the form
            # now shows.
            _analysis.save_analysis(active_test, result, inputs)
            st.session_state.project.touch()
            st.session_state.saved_result = result
            st.rerun()

        # ── Step 4b: PDF Report download ──────────────────────────────────────
        st.divider()
        st.subheader("Step 4b — Download Report")
        if st.button("📄 Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF…"):
                try:
                    # The service picks the radial coordinate the method reports
                    # (r_w at the pumped well, the observation distance
                    # otherwise) and names the file.
                    pdf_bytes, fname = _analysis.report_pdf(
                        result, inputs, st.session_state.formatted_df,
                        project_info=st.session_state.project.info,
                        test_label=_active_test().name,
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

        spec_r = _registry.require(method_r)

        with plot_tab1:
            st.plotly_chart(_analysis.diagnostic_plot(result, inputs),
                            use_container_width=True)

        with plot_tab2:
            st.markdown("**Adjust T and S with the sliders — the curve updates in real time.**")
            st.markdown("Match the orange curve to the observed data, then compare with the automated fit (red dashed).")

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
                if "S" in spec_r.parameters:
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
                else:
                    S_manual = result.S
                    st.caption(f"{spec_r.label} does not estimate S, so only T "
                               "is adjusted.")

            fig_m, rmse_m = _analysis.manual_match(
                result, inputs, T_manual, S_manual)

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


# ─────────────────────────────────────────────────────────────────────────────
# Saved analyses for the active test — shown whether or not a fit is on screen
# ─────────────────────────────────────────────────────────────────────────────
with tab_calc:
    _test = _active_test()
    _frame = _ws.comparison_frame(_test)
    if not _frame.empty:
        st.divider()
        st.subheader(f"Compare saved analyses — {_test.name}")
        st.dataframe(_frame, hide_index=True, use_container_width=True)
        st.caption(
            "Ranked by AICc, which charges for every extra parameter — not by R², "
            "which always favours the most complex method. ΔAICc under 2 means the "
            "data cannot tell two methods apart; over 10 is decisive. Weight is the "
            "relative likelihood that each method is the best of this set."
        )
        _stale = _ws.stale_count(_test)
        if _stale:
            st.caption(f"{_stale} earlier analysis(es) were run on different "
                       "readings and are left out of this comparison.")
