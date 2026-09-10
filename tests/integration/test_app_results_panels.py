"""The workspace results panel must render for every method.

Three methods shipped whose results panel crashed after a successful fit:
Hantush-Jacob, Boulton and Warren-Root registered no plot adapters, and the panel
calls them. The existing UI smoke tests only ever rendered a Theis result, so
nothing exercised the code path that broke.

These render the real app.py with a real fitted result for every registered
method — the panel, the diagnostic plot, the manual-matching sliders — and check
that methods estimating more than T and S show those extra parameters, which the
four-metric row cannot.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.calculator import registry
from src.calculator.base import CalculationInput

pytestmark = pytest.mark.slow

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(Path(__file__).resolve().parents[2] / "app.py")
TIMEOUT = 240

Q_OBS, R_OBS = 500 / 86400, 50.0
Q_PW, R_W, R_C = 300 / 86400, 0.30, 0.20

# The symbol each multi-parameter method must show beyond T and S.
EXTRA_SYMBOL = {
    "hantush_jacob_1955": "B",
    "boulton_1963": "S_y",
    "warren_root_1963": "ω",
}


def _app_with_result(spec):
    if spec.data_source == registry.PUMPED_WELL:
        df = pd.read_csv("tests/papadopulos_cooper_sample.csv")
        result = spec.build().calculate(
            CalculationInput(df=df, Q=Q_PW, r=R_W, r_w=R_W, r_c=R_C))
        inputs = {"Q": Q_PW, "r": R_W, "r_w": R_W, "r_c": R_C}
    else:
        df = pd.read_csv("tests/theis_cooper_jacob_sample.csv")
        result = spec.build().calculate(CalculationInput(df=df, Q=Q_OBS, r=R_OBS))
        inputs = {"Q": Q_OBS, "r": R_OBS, "r_w": 0.10, "r_c": 0.10}

    assert result.success, result.error_message

    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.session_state.calc_result = result
    at.session_state.formatted_df = df
    at.session_state.calc_inputs = {**inputs, "method": spec.label, "n_excluded": 0}
    at.run()
    return at, result


@pytest.mark.parametrize("spec", registry.all_specs(), ids=lambda s: s.id)
def test_results_panel_renders_without_error(spec):
    at, _ = _app_with_result(spec)
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


@pytest.mark.parametrize("spec", registry.all_specs(), ids=lambda s: s.id)
def test_the_fitted_transmissivity_is_shown(spec):
    at, result = _app_with_result(spec)
    assert f"{result.T_day:.2f}" in [m.value for m in at.metric]


@pytest.mark.parametrize("method_id", sorted(EXTRA_SYMBOL))
def test_parameters_beyond_t_and_s_are_listed(method_id):
    """The four-metric row shows only T and S; the rest must appear somewhere."""
    at, _ = _app_with_result(registry.get(method_id))

    symbols = set()
    for frame in at.dataframe:
        df = frame.value
        if "Parameter" in df.columns:
            symbols |= set(df["Parameter"])
    assert EXTRA_SYMBOL[method_id] in symbols, symbols


def test_original_methods_do_not_show_a_redundant_table():
    """T and S alone are already in the metric row."""
    at, _ = _app_with_result(registry.get("theis_1935"))

    assert not any("Parameter" in f.value.columns for f in at.dataframe)
