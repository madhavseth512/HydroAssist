"""Glue between the Streamlit workspace and the project model.

Kept free of Streamlit so it can be tested directly; app.py only wires these into
widgets. The Step 2 widget keys live here too, because they are the contract
between what a saved project contains and what the form shows when it reopens.
"""

from typing import Mapping, Optional

import pandas as pd

from src.project.model import Project, PumpingTest, TestSetup

Q_TO_SI = {"m³/day": 1 / 86400, "m³/hour": 1 / 3600, "m³/s": 1.0, "L/s": 1e-3}
Q_UNITS = list(Q_TO_SI)

# Step 2 widget keys in app.py.
KEY_Q = "q_input"
KEY_Q_UNIT = "q_unit"
KEY_R = "r_input"
KEY_RW = "rw_input"
KEY_RC = "rc_input"
KEY_TEST_TYPE = "test_type"
KEY_TP = "tp_hours"
KEY_CLOCK = "recovery_clock"

TEST_TYPE_LABELS = {
    "constant_rate": "Constant-rate pumping",
    "recovery": "Recovery (after the pump stops)",
}
CLOCK_LABELS = {
    "since_stop": "Time since pumping stopped (t′)",
    "since_start": "Time since pumping started (t)",
}

DEFAULTS = {KEY_Q: 500.0, KEY_Q_UNIT: "m³/day", KEY_R: 50.0,
            KEY_RW: 0.15, KEY_RC: 0.10,
            KEY_TEST_TYPE: "constant_rate", KEY_TP: 24.0, KEY_CLOCK: "since_stop"}

# The min/max each widget declares. A reopened value outside them would make
# Streamlit raise when the widget renders, so values are clamped on the way in.
BOUNDS = {KEY_Q: (0.0001, 100000.0), KEY_R: (0.1, 5000.0),
          KEY_RW: (0.01, 5.0), KEY_RC: (0.01, 5.0), KEY_TP: (0.01, 10000.0)}


def q_to_si(value: float, unit: str) -> float:
    """Pumping rate in the entered unit to m³/s."""
    if unit not in Q_TO_SI:
        raise ValueError(f"Unknown pumping-rate unit {unit!r}; "
                         f"expected one of {', '.join(Q_UNITS)}.")
    return float(value) * Q_TO_SI[unit]


def _clamp(key: str, value: float) -> float:
    lo, hi = BOUNDS[key]
    return float(min(max(float(value), lo), hi))


def setup_from_inputs(values: Mapping) -> TestSetup:
    """The Step 2 form as a TestSetup. Missing keys fall back to the defaults —
    Streamlit drops the state of widgets not rendered in a run, which happens to
    the well-radius inputs whenever the method does not need them."""
    unit = values.get(KEY_Q_UNIT, DEFAULTS[KEY_Q_UNIT])
    q = float(values.get(KEY_Q, DEFAULTS[KEY_Q]))
    test_type = values.get(KEY_TEST_TYPE, DEFAULTS[KEY_TEST_TYPE])
    return TestSetup(
        Q_m3s=q_to_si(q, unit), Q_entered=q, Q_unit=unit,
        r_m=float(values.get(KEY_R, DEFAULTS[KEY_R])),
        r_w_m=float(values.get(KEY_RW, DEFAULTS[KEY_RW])),
        r_c_m=float(values.get(KEY_RC, DEFAULTS[KEY_RC])),
        test_type=test_type,
        # Only meaningful for a recovery test; stored in seconds.
        t_p_s=(float(values.get(KEY_TP, DEFAULTS[KEY_TP])) * 3600.0
               if test_type == "recovery" else None),
        recovery_clock=values.get(KEY_CLOCK, DEFAULTS[KEY_CLOCK]),
    )


def setup_from_run(calc_inputs: Mapping) -> TestSetup:
    """The setup an analysis was ACTUALLY run with.

    Recorded from the run rather than read back from the form at save time: the
    user may have edited Q after running, and an analysis stored with the wrong Q
    would misreport the test it describes.
    """
    return TestSetup(
        Q_m3s=calc_inputs.get("Q"),
        Q_entered=calc_inputs.get("Q_entered"),
        Q_unit=calc_inputs.get("Q_unit", DEFAULTS[KEY_Q_UNIT]),
        r_m=float(calc_inputs.get("r", DEFAULTS[KEY_R])),
        r_w_m=float(calc_inputs.get("r_w", DEFAULTS[KEY_RW])),
        r_c_m=float(calc_inputs.get("r_c", DEFAULTS[KEY_RC])),
        test_type=calc_inputs.get("test_type", DEFAULTS[KEY_TEST_TYPE]),
        t_p_s=calc_inputs.get("t_p_s"),
        recovery_clock=calc_inputs.get("recovery_clock", DEFAULTS[KEY_CLOCK]),
    )


def inputs_from_setup(setup: TestSetup) -> dict:
    """Widget values that reproduce `setup` in the Step 2 form."""
    unit = setup.Q_unit if setup.Q_unit in Q_TO_SI else DEFAULTS[KEY_Q_UNIT]
    if setup.Q_entered is not None and setup.Q_unit in Q_TO_SI:
        q = setup.Q_entered
    elif setup.Q_m3s is not None:
        # Re-express the stored SI rate in the unit being shown.
        q = setup.Q_m3s / Q_TO_SI[unit]
    else:
        q = DEFAULTS[KEY_Q]
    return {
        KEY_Q: _clamp(KEY_Q, q),
        KEY_Q_UNIT: unit,
        KEY_R: _clamp(KEY_R, setup.r_m),
        KEY_RW: _clamp(KEY_RW, setup.r_w_m),
        KEY_RC: _clamp(KEY_RC, setup.r_c_m),
        KEY_TEST_TYPE: (setup.test_type if setup.test_type in TEST_TYPE_LABELS
                        else DEFAULTS[KEY_TEST_TYPE]),
        KEY_TP: (_clamp(KEY_TP, setup.t_p_s / 3600.0) if setup.t_p_s
                 else DEFAULTS[KEY_TP]),
        KEY_CLOCK: (setup.recovery_clock if setup.recovery_clock in CLOCK_LABELS
                    else DEFAULTS[KEY_CLOCK]),
    }


def new_project() -> Project:
    """A blank project with one empty test, which is what the workspace starts in."""
    project = Project()
    project.add_test()
    return project


def project_summary(project: Project) -> str:
    n_tests, n_analyses = len(project.tests), project.analysis_count
    return (f"{n_tests} test{'s' if n_tests != 1 else ''} · "
            f"{n_analyses} saved analys{'es' if n_analyses != 1 else 'is'}")


def stale_count(test: PumpingTest) -> int:
    """Analyses kept on this test but run against readings it no longer holds."""
    return sum(1 for a in test.analyses if test.is_stale(a))


def evidence(delta: Optional[float]) -> str:
    """Burnham & Anderson's conventional reading of ΔAICc.

    Public because the HTTP API reports the same wording; duplicating the
    thresholds would let the two drift.
    """
    if delta is None:
        return "not ranked"
    if delta < 1e-9:
        return "best"
    if delta < 2:
        return "substantial support"
    if delta < 10:
        return "less support"
    return "essentially no support"


def comparison_frame(test: PumpingTest) -> pd.DataFrame:
    """The test's current analyses as a display table, best model first."""
    rows = test.comparison()
    if not rows:
        return pd.DataFrame()

    def fmt(value, spec):
        return "—" if value is None else format(value, spec)

    return pd.DataFrame([
        {
            "Rank": rank,
            "Method": r["method"],
            "Parameters": fmt(r["n_params"], "d"),
            "T (m²/day)": fmt(r["T_m2_per_day"], ".2f"),
            "S": fmt(r["S"], ".2e"),
            "R²": fmt(r["r_squared"], ".4f"),
            "AICc": fmt(r["aicc"], ".1f"),
            "ΔAICc": fmt(r["delta_aicc"], ".1f"),
            "Weight": fmt(r["akaike_weight"], ".2f"),
            "Evidence": evidence(r["delta_aicc"]),
        }
        for rank, r in enumerate(rows, 1)
    ])
