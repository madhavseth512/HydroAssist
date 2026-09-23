"""The methods against real pumping tests, with the answers their analysts published.

`test_published_values.py` checks the models against tabulated well functions —
mathematics, computed exactly. This file is the other half: measurements from
real wells, with all the noise, partial penetration and boundary effects that
implies, analysed by a hydrogeologist who published what he got.

The data is from Walton (1962), Illinois State Water Survey Bulletin 49, which
is `knowledge source/General/ISWSB-49.pdf` in this repository. The tables are in
its text; his computed parameters are printed inside figure images, and were
recovered with the OCR built for the scanned papers.

**These tolerances are wide on purpose.** Walton matched type curves by eye on
log-log paper. Least squares on the same points is a different estimator, and
where the two disagree it is not obvious that the older one is wrong — so where
the gap is large, the test asserts something stronger instead: that our
parameters explain his measurements at least as well as his own do, measured as
RMSE against the readings. On the Dieterich test ours is 3.8 cm against his
5.5 cm.

What is deliberately *not* asserted: that we reproduce his numbers closely. Two
estimators on eleven noisy points need not agree, and pretending otherwise would
make this a test of coincidence.
"""

import numpy as np
import pandas as pd
import pytest

from src.calculator.base import CalculationInput
from src.calculator.cooper_jacob import CooperJacobCalculator
from src.calculator.methods.hantush_jacob import (
    HantushJacobCalculator, hantush_jacob_drawdown,
)
from src.calculator.theis import TheisCalculator

# Walton works in US customary units throughout.
GPM_TO_M3S = 0.003785411784 / 60.0
FT = 0.3048
GPD_FT_TO_M2DAY = 0.012419          # gallon/day/foot -> m^2/day

FIELD_DATA = "tests/fixtures/field_data"


def _load(name: str) -> pd.DataFrame:
    """A published table as the calculators want it: seconds and metres."""
    frame = pd.read_csv(f"{FIELD_DATA}/{name}.csv", comment="#")
    return pd.DataFrame({"time_s": frame.time_min * 60.0,
                         "drawdown_m": frame.drawdown_ft * FT})


def _gpd_ft(t_m2_day: float) -> float:
    return t_m2_day / GPD_FT_TO_M2DAY


class TestGridleyConfinedTest:
    """Gridley, Illinois. 220 gpm, non-leaky artesian.

    Walton analysed observation well 1 with the non-leaky artesian type curve —
    which is the Theis method — and the pumped well with a straight line, which
    is Cooper-Jacob. He published an average T of 11,000 gpd/ft and S of
    2.2e-5, and his figure 26 shows 12,100 gpd/ft from the straight line alone.
    """

    Q = 220 * GPM_TO_M3S
    R = 824 * FT
    PUBLISHED_T = 11_000          # gpd/ft, averaged over both analyses
    PUBLISHED_S = 2.2e-5
    PUBLISHED_T_STRAIGHT_LINE = 12_100    # his figure 26, pumped well

    @pytest.fixture(scope="class")
    def theis_fit(self):
        return TheisCalculator().calculate(
            CalculationInput(df=_load("gridley_well1"), Q=self.Q, r=self.R))

    def test_theis_fits_the_measurements(self, theis_fit):
        assert theis_fit.success, theis_fit.error_message
        assert theis_fit.r_squared > 0.99

    def test_theis_transmissivity_is_close_to_what_walton_published(self, theis_fit):
        """9,900-ish against his 11,000 average. A tenth is what two estimators
        differ by on one well's readings."""
        assert _gpd_ft(theis_fit.T_day) == pytest.approx(self.PUBLISHED_T, rel=0.15)

    def test_theis_storativity_is_the_right_order_and_close(self, theis_fit):
        assert theis_fit.S == pytest.approx(self.PUBLISHED_S, rel=0.25)

    def test_the_straight_line_matches_his_straight_line(self):
        """The closest comparison available anywhere in this suite: the same
        method, on the same readings, against the number he printed."""
        fit = CooperJacobCalculator().calculate(
            CalculationInput(df=_load("gridley_well3"), Q=self.Q, r=0.5 * FT))

        assert fit.success, fit.error_message
        assert _gpd_ft(fit.T_day) == pytest.approx(
            self.PUBLISHED_T_STRAIGHT_LINE, rel=0.05)

    @pytest.mark.parametrize("r_ft", [0.5, 1.0, 2.0])
    def test_transmissivity_from_a_pumped_well_does_not_depend_on_its_radius(self, r_ft):
        """Cooper-Jacob takes T from the slope, which carries no r. This is why
        Walton could report T from the pumping well at all."""
        fit = CooperJacobCalculator().calculate(
            CalculationInput(df=_load("gridley_well3"), Q=self.Q, r=r_ft * FT))

        assert _gpd_ft(fit.T_day) == pytest.approx(
            self.PUBLISHED_T_STRAIGHT_LINE, rel=0.05)

    def test_but_storativity_from_a_pumped_well_does(self):
        """And this is why he refused to report S from it. The intercept
        carries r^2, so S moves with the radius we assume — a value we do not
        know for this well."""
        fits = [CooperJacobCalculator().calculate(
                    CalculationInput(df=_load("gridley_well3"), Q=self.Q, r=r * FT))
                for r in (0.5, 2.0)]

        assert fits[0].S / fits[1].S == pytest.approx(16.0, rel=0.1)


class TestDieterichLeakyTest:
    """Dieterich, Illinois. 25 gpm for 20 hours, leakage through a confining bed.

    Walton matched the leaky artesian type curves, interpolating r/B = 0.22, and
    published T = 1,500 gpd/ft, S = 0.0002 and a vertical permeability of
    0.10 gpd/sq ft.
    """

    Q = 25 * GPM_TO_M3S
    R = 96 * FT
    PUBLISHED_T = 1_500
    PUBLISHED_S = 2e-4
    PUBLISHED_R_OVER_B = 0.22

    @pytest.fixture(scope="class")
    def data(self):
        return _load("dieterich_well19")

    @pytest.fixture(scope="class")
    def leaky_fit(self, data):
        return HantushJacobCalculator().calculate(
            CalculationInput(df=data, Q=self.Q, r=self.R))

    def test_it_fits(self, leaky_fit):
        assert leaky_fit.success, leaky_fit.error_message
        assert leaky_fit.r_squared > 0.99

    def test_the_parameters_are_the_right_size(self, leaky_fit):
        """Wider than Gridley, and the next test says why that is acceptable:
        visual matching trades T against r/B, so two answers can differ by a
        quarter while describing the same curve."""
        assert _gpd_ft(leaky_fit.T_day) == pytest.approx(self.PUBLISHED_T, rel=0.30)
        assert leaky_fit.S == pytest.approx(self.PUBLISHED_S, rel=0.30)
        assert self.R / leaky_fit.value("B") == pytest.approx(
            self.PUBLISHED_R_OVER_B, rel=0.40)

    def test_our_answer_explains_his_data_better_than_his_own(self, data, leaky_fit):
        """The test that makes the looser tolerances above defensible. If our
        parameters were wrong rather than merely different, they would fit his
        readings worse."""
        time_s = data.time_s.to_numpy()
        observed = data.drawdown_m.to_numpy()

        def rmse(t_day, storativity, r_over_b):
            predicted = hantush_jacob_drawdown(
                time_s, t_day / 86400.0, storativity, self.R / r_over_b,
                self.Q, self.R)
            return float(np.sqrt(np.mean((predicted - observed) ** 2)))

        walton = rmse(self.PUBLISHED_T * GPD_FT_TO_M2DAY, self.PUBLISHED_S,
                      self.PUBLISHED_R_OVER_B)
        ours = rmse(leaky_fit.T_day, leaky_fit.S, self.R / leaky_fit.value("B"))

        assert ours < walton
        assert ours < 0.05          # under 5 cm on drawdowns up to 2 m

    def test_the_leaky_model_beats_theis_on_leaky_data(self, data):
        """The conclusion Walton reached by eye, reached by arithmetic: this
        test is leaky, and a confined model does not describe it."""
        confined = TheisCalculator().calculate(
            CalculationInput(df=data, Q=self.Q, r=self.R))
        leaky = HantushJacobCalculator().calculate(
            CalculationInput(df=data, Q=self.Q, r=self.R))

        assert leaky.r_squared > confined.r_squared
        assert leaky.to_state_dict()["gof"]["aicc"] < confined.to_state_dict()["gof"]["aicc"]

    def test_theis_overestimates_transmissivity_here(self, data):
        """Ignoring leakage inflates T badly — roughly double. It is the error
        this method exists to prevent, and worth pinning."""
        confined = TheisCalculator().calculate(
            CalculationInput(df=data, Q=self.Q, r=self.R))

        assert _gpd_ft(confined.T_day) > 1.5 * self.PUBLISHED_T


class TestTheFixturesThemselves:
    """A transcription error would look exactly like a fitting error."""

    @pytest.mark.parametrize("name,rows,q_gpm", [
        ("gridley_well1", 22, 220),
        ("gridley_well3", 14, 220),
        ("dieterich_well19", 11, 25),
    ])
    def test_the_tables_are_intact(self, name, rows, q_gpm):
        frame = pd.read_csv(f"{FIELD_DATA}/{name}.csv", comment="#")

        assert len(frame) == rows
        assert list(frame.columns) == ["time_min", "drawdown_ft"]
        assert frame.time_min.is_monotonic_increasing
        assert (frame.drawdown_ft > 0).all()

    @pytest.mark.parametrize("name", ["gridley_well1", "gridley_well3",
                                      "dieterich_well19"])
    def test_drawdown_only_grows(self, name):
        """Every one of these is a constant-rate test, so drawdown increases
        throughout. A pair read from the wrong row would usually break this."""
        frame = pd.read_csv(f"{FIELD_DATA}/{name}.csv", comment="#")

        assert frame.drawdown_ft.is_monotonic_increasing

    @pytest.mark.parametrize("name", ["gridley_well1", "gridley_well3",
                                      "dieterich_well19"])
    def test_each_file_records_where_it_came_from(self, name):
        """These numbers are only evidence if a reader can find them in the
        source."""
        header = open(f"{FIELD_DATA}/{name}.csv", encoding="utf-8").read()

        assert "Walton" in header and "Bulletin 49" in header
        assert "Table" in header and "gpm" in header
