"""Running an analysis, and everything that hangs off a finished one.

`app.py` held its own copies of: which calculator to build and with what
geometry, which methods to offer for a dataset, what to keep beside a result,
how to record it against a project, and how to assemble the PDF. None of that is
Streamlit-specific, but living inside the Streamlit script made it unreachable
from anywhere else.

The registry decides dispatch, `TestSetup` carries the test parameters, and the
project model records the outcome — this module only joins them up.
"""

from typing import Optional, Sequence

import pandas as pd

from src.calculator import registry
from src.calculator.base import CalculationResult
from src.project import workspace as ws
from src.project.model import AnalysisRecord, PumpingTest, TestSetup
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ── Running a fit ─────────────────────────────────────────────────────────────

def build_inputs(df: pd.DataFrame, setup: TestSetup, spec) -> "CalculationInput":
    """The CalculationInput a method needs for this setup.

    Two things vary by method rather than by user choice, and both come from the
    registry entry: a method fitted at the pumped well face takes the screen
    radius as its radial coordinate rather than an observation distance, and a
    recovery method needs the discharge history as a schedule ending at zero.
    """
    from src.calculator.base import CalculationInput
    from src.calculator.geometry import PumpingSchedule, RateStep

    extra = {}
    if setup.test_type == registry.RECOVERY:
        if not setup.t_p_s:
            raise ValueError(
                "A recovery analysis needs the pumping duration t_p — how long "
                "the well pumped before it was switched off."
            )
        # The pump ran at Q for t_p, then stopped.
        extra = dict(
            test_type=registry.RECOVERY,
            schedule=PumpingSchedule(steps=(RateStep(0.0, setup.Q_m3s),
                                            RateStep(float(setup.t_p_s), 0.0))),
            options={"recovery_clock": setup.recovery_clock},
        )

    if spec.data_source == registry.PUMPED_WELL:
        return CalculationInput(df=df, Q=setup.Q_m3s, r=setup.r_w_m,
                                r_w=setup.r_w_m, r_c=setup.r_c_m, **extra)
    return CalculationInput(df=df, Q=setup.Q_m3s, r=setup.r_m, **extra)


def run_analysis(df: pd.DataFrame, method: str,
                 setup: TestSetup) -> CalculationResult:
    """Fit `method` (an id or a label) to `df` under `setup`."""
    spec = registry.require(method)
    return spec.build().calculate(build_inputs(df, setup, spec))


def run_record(setup: TestSetup, method: str, result: CalculationResult,
               n_points: Optional[int] = None) -> dict:
    """What to keep beside a result: the setup it actually ran with.

    Held separately from the live form because the user may edit the form
    afterwards, and an analysis described by the wrong Q is worse than none.
    `Q_day` and `r_c` also feed the validator's method pre-check on the next
    upload.
    """
    spec = registry.require(method)
    return {
        "Q": setup.Q_m3s,
        "r": setup.r_m,
        "method": spec.label,
        "method_id": spec.id,
        "n_excluded": (int(result.diagnostics.get("n_excluded", 0))
                       if result.success else 0),
        "r_w": setup.r_w_m,
        "r_c": setup.r_c_m,
        "Q_day": (setup.Q_m3s or 0.0) * 86400.0,
        "Q_entered": setup.Q_entered,
        "Q_unit": setup.Q_unit,
        "test_type": setup.test_type,
        "t_p_s": setup.t_p_s,
        "recovery_clock": setup.recovery_clock,
        "n_points": n_points,
    }


def chat_record(record: dict, n_points: Optional[int] = None) -> dict:
    """The slice of the record the conversation is told about."""
    return {
        "Q": record.get("Q"), "r": record.get("r"),
        "method": record.get("method"),
        "r_w": record.get("r_w"), "r_c": record.get("r_c"),
        "n_points": n_points if n_points is not None else record.get("n_points"),
    }


# ── Choosing a method ─────────────────────────────────────────────────────────

def method_options(validation, test_type: str = registry.CONSTANT_RATE) -> tuple:
    """(offered, gated_out) for a test type, filtered by the validator's
    suitability check. The first method for the test type is always offered, so
    the list is never empty."""
    for_type = [spec.label for spec in registry.for_test_type(test_type)]
    if validation is None or not validation.suitable_methods:
        return for_type, []
    suitable = [m for m in for_type if m in validation.suitable_methods]
    if not suitable:
        suitable = for_type[:1]
    gated_out = [m for m in for_type if m not in suitable]
    return suitable, gated_out


def gating_reasons(validation, gated_out: Sequence[str]) -> dict:
    """Why each excluded method was excluded, as far as the validator says.

    The exclusions are the part that teaches: "not applicable because you gave
    no casing radius" tells a practitioner something about the method, while a
    greyed name tells them nothing. The validator records its reasoning as
    prose in `info` and `warnings` without attaching it to a method, so the
    sentence mentioning a method is matched back to it here.

    A method with nothing said about it gets a general sentence rather than a
    guess. Inventing a specific reason would be worse than admitting none.
    """
    if not gated_out:
        return {}

    said = list(getattr(validation, "info", []) or []) +         list(getattr(validation, "warnings", []) or [])

    reasons = {}
    for label in gated_out:
        spec = registry.get(label)
        names = {label, label.split(" (")[0]}
        if spec:
            names.update(spec.aliases)

        explanation = next(
            (sentence for sentence in said
             if any(name.lower() in sentence.lower() for name in names)), None)
        reasons[label] = explanation or (
            "This data set does not meet what the method needs — see the data "
            "notes in Step 1.")
    return reasons


def method_help(options: Sequence[str]) -> str:
    """Explanations assembled from the registry, so a new method arrives with
    its own rather than silently having none."""
    lines = []
    for label in options:
        spec = registry.get(label)
        if spec and spec.ui_help:
            lines.append(f"{spec.label}: {spec.ui_help}")
    return "\n\n".join(lines)


def same_method(a: str, b: str) -> bool:
    """Loose match between a canonical label and a method name from the LLM.

    The Manager may return "Theis", "theis method" or "Theis (1935)"; dropping
    years, punctuation and spacing reduces "Papadopulos-Cooper (1967)" and
    "papadopulos cooper" to the same key.
    """
    def key(s):
        return "".join(ch for ch in s.lower() if ch.isalpha())

    ka, kb = key(a), key(b)
    if not ka or not kb:
        return False
    return ka.startswith(kb) or kb.startswith(ka)


def default_method_index(options: Sequence[str], suggested: str = "") -> int:
    """Preselect the method the consultant suggested, when it is offered."""
    if suggested:
        for i, option in enumerate(options):
            if same_method(option, suggested):
                return i
    return 0


# ── A finished result ─────────────────────────────────────────────────────────

def estimates_S(result: CalculationResult) -> bool:
    """False for a method that reports no storativity, such as Theis recovery,
    whose `S` field is only the unset default."""
    return result.get("S") is not None


def format_ci(ci, scale: float = 86400.0, unit: str = "m²/day") -> str:
    if ci is None:
        return "not available"
    return f"[{ci[0] * scale:.2f}, {ci[1] * scale:.2f}] {unit}"


def plot_context(record: dict) -> registry.PlotContext:
    return registry.PlotContext(
        Q=record.get("Q"),
        r=record.get("r") or 0.0,
        r_w=record.get("r_w") or 0.10,
        r_c=record.get("r_c") or 0.10,
        n_excluded=int(record.get("n_excluded") or 0),
    )


def _spec_for(result: CalculationResult, record: dict):
    return registry.require(record.get("method") or result.method_id
                            or result.method)


def diagnostic_plot(result: CalculationResult, record: dict):
    return _spec_for(result, record).diagnostic_plot(result, plot_context(record))


def manual_match(result: CalculationResult, record: dict,
                 T_manual: float, S_manual: float) -> tuple:
    """(figure, RMSE in mm) for hand-matched parameters. Every parameter other
    than T and S stays at its fitted value."""
    return _spec_for(result, record).interactive_plot(
        result, plot_context(record), T_manual, S_manual)


def save_analysis(test: PumpingTest, result: CalculationResult,
                  record: dict) -> AnalysisRecord:
    """Record a result against a project test, with the setup it ran with."""
    spec = registry.get(record.get("method", "")) or registry.get(result.method)
    return test.add_analysis(
        result, result.method,
        spec.id if spec else result.method_id,
        setup=ws.setup_from_run(record),
    )


def restore_last_analysis(test: PumpingTest):
    """The test's most recent saved analysis, fitted again so it can be shown.

    Opening a test used to show nothing, because a result lived for one session
    and one test. Its readings and its saved analyses are in the project file, so
    what a reopened test is missing is only the fit itself — and a fit is
    deterministic: the same method under the same setup on the same readings
    reproduces it exactly, in milliseconds.

    Re-running rather than storing the result whole is deliberate. The curve, the
    residuals and the figures are derived from the readings, which are already
    here; keeping a copy of them per analysis would multiply the size of every
    project file for numbers that can be recomputed.

    Returns `(result, record)`, or `(None, {})` when there is nothing to restore —
    no saved analysis, one that describes replaced readings, or a method this
    version no longer has. None of those is an error; the panel is simply empty.
    """
    if test.readings is None:
        return None, {}

    current = [record for record in test.analyses
               if not test.is_stale(record) and record.success]
    if not current:
        return None, {}

    saved = current[-1]
    try:
        setup = saved.setup
        result = run_analysis(test.readings.to_frame(),
                              saved.method_id or saved.method_label, setup)
        if not result.success:
            return None, {}
        return result, run_record(setup, saved.method_id or saved.method_label,
                                  result, n_points=len(test.readings))
    except Exception as error:                              # noqa: BLE001
        logger.warning(f"Could not restore {saved.method_label} on "
                       f"{test.name}: {error}")
        return None, {}


def report_filename(method_label: str) -> str:
    return (method_label.split("(")[0].strip().lower().replace(" ", "_")
            + "_report.pdf")


def report_pdf(result: CalculationResult, record: dict, df: pd.DataFrame,
               project_info=None, test_label: str = "") -> tuple:
    """(pdf bytes, filename). A method fitted at the pumped well reports its
    screen radius rather than an observation distance."""
    from src.calculator.report import generate_pdf

    spec = _spec_for(result, record)
    at_well = spec.data_source == registry.PUMPED_WELL
    pdf = generate_pdf(
        result,
        record.get("Q"),
        record.get("r_w") if at_well else record.get("r"),
        spec.label,
        df,
        int(record.get("n_excluded") or 0),
        r_c=record.get("r_c") if at_well else None,
        project_info=project_info,
        test_label=test_label,
    )
    return pdf, report_filename(spec.label)
