"""Golden-number regression guard for the calculator architecture migration.

`tests/fixtures/golden_fits.json` pins the exact T, S, RMSE and R-squared that the
three original calculators produce on the two bundled sample CSVs, recorded before
any refactoring began.

This is the safety net for the whole migration. Several planned steps are supposed
to be numerically inert:

  * extracting and vectorising the Stehfest inverter, and moving from
    scipy.special.kv to the exponentially-scaled kve
  * porting each calculator onto the shared CurveFitCalculator template
  * introducing FittedParameter and the generalised CalculationResult

If any of those changes a fitted value, it changed the optimisation rather than
just moving code, and that is a bug. The tolerances below are deliberately far
tighter than the physical accuracy of the methods — they are testing that the
refactor was a no-op, not that the hydrogeology is right. The paper-validated
accuracy tests live in test_theis.py, test_cooper_jacob.py and
test_papadopulos_cooper.py.

To re-baseline after an INTENTIONAL numerical change, regenerate the fixture and
say plainly in the commit message which values moved and why.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.calculator.base import CalculationInput
from src.calculator.cooper_jacob import CooperJacobCalculator
from src.calculator.papadopulos_cooper import PapadopulosCooperCalculator
from src.calculator.theis import TheisCalculator

GOLDEN_PATH = Path("tests/fixtures/golden_fits.json")

# A pure code move must reproduce the fit to near machine precision.
RTOL = 1e-9


def _golden() -> dict:
    data = json.loads(GOLDEN_PATH.read_text())
    # Metadata about intentional re-baselines, not a method entry.
    data.pop("_provenance", None)
    return data


CALCULATORS = {
    "theis_1935": TheisCalculator,
    "cooper_jacob_1946": CooperJacobCalculator,
    "papadopulos_cooper_1967": PapadopulosCooperCalculator,
}


def _rerun(method_id: str, setup: dict):
    df = pd.read_csv(Path("tests") / setup["sample"])
    kwargs = {k: v for k, v in setup.items() if k != "sample"}
    return CALCULATORS[method_id]().calculate(CalculationInput(df=df, **kwargs))


@pytest.fixture(scope="module")
def golden():
    return _golden()


class TestGoldenFixture:

    def test_fixture_exists_and_covers_every_calculator(self, golden):
        assert set(golden) == set(CALCULATORS), (
            "Every shipped calculator needs a golden baseline before it is refactored"
        )

    def test_fixture_records_the_test_setup(self, golden):
        """Without the setup the numbers cannot be reproduced."""
        for method_id, g in golden.items():
            assert "sample" in g["setup"], method_id
            assert "Q" in g["setup"], method_id


@pytest.mark.parametrize("method_id", sorted(CALCULATORS))
class TestFitsAreUnchanged:

    def test_transmissivity_unchanged(self, method_id, golden):
        result = _rerun(method_id, golden[method_id]["setup"])
        assert result.success
        assert result.T == pytest.approx(golden[method_id]["T"], rel=RTOL)

    def test_storativity_unchanged(self, method_id, golden):
        result = _rerun(method_id, golden[method_id]["setup"])
        assert result.S == pytest.approx(golden[method_id]["S"], rel=RTOL)

    def test_goodness_of_fit_unchanged(self, method_id, golden):
        result = _rerun(method_id, golden[method_id]["setup"])
        g = golden[method_id]
        assert result.rmse == pytest.approx(g["rmse"], rel=RTOL)
        assert result.r_squared == pytest.approx(g["r_squared"], rel=RTOL)

    def test_confidence_intervals_unchanged(self, method_id, golden):
        """The CI comes from the covariance matrix, so it is the most sensitive
        indicator that an optimiser setting drifted."""
        result = _rerun(method_id, golden[method_id]["setup"])
        g = golden[method_id]

        for name, actual, expected in [("T_ci", result.T_ci, g["T_ci"]),
                                       ("S_ci", result.S_ci, g["S_ci"])]:
            if expected is None:
                assert actual is None, f"{method_id}: {name} became computable"
            else:
                assert actual is not None, f"{method_id}: {name} was lost"
                assert list(actual) == pytest.approx(expected, rel=RTOL)

    def test_derived_day_units_consistent(self, method_id, golden):
        result = _rerun(method_id, golden[method_id]["setup"])
        assert result.T_day == pytest.approx(result.T * 86400.0, rel=1e-12)

    def test_validity_note_count_unchanged(self, method_id, golden):
        """Notes are paper-sourced interpretation; silently losing one during a
        refactor would remove a warning the user is relied upon to see."""
        result = _rerun(method_id, golden[method_id]["setup"])
        assert len(result.validity_notes) == golden[method_id]["n_validity_notes"]
