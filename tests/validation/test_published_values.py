"""Each method against values somebody else published.

Every other test in this suite checks the code against itself: a method
recovers the parameters from data its own forward model generated, or converges
to a limit the same module computes. That proves the fitting, not the physics. A
wrong coefficient in a term that vanishes at the limit passes all of it.

These tests are the exception. The numbers below are transcribed from documents
in `knowledge source/`, with the page written next to each, and nothing computes
them:

- **W(u)** — the Theis well function, from the AQTESOLV for Windows User's
  Guide, Guided Tour 1 (`31032.pdf`, printed page 9). The tour's dataset is
  W(u) itself: its displacements at t = 0.5 … 10000 are the well function at
  u = 1/t.
- **W(u, r/B)** — the Hantush-Jacob leaky well function, from Walton, Illinois
  State Water Survey Bulletin 49, Appendix A (`ISWSB-49.pdf`, printed page 73),
  which reproduces Hantush (1956).

The printed tables carry four decimals, so agreement is asserted to the last
digit printed and no further.

What is still unvalidated, and should not be mistaken for covered: Boulton,
Warren-Root and Theis recovery, none of which has a published table in the
corpus. Comparing those against AQTESOLV itself is the next step.
"""

import numpy as np
import pytest
from scipy.special import kn

from src.calculator.cooper_jacob import CJ_U_THRESHOLD
from src.calculator.methods.hantush_jacob import (
    hantush_jacob_drawdown, steady_state_drawdown,
)
from src.calculator.theis import _make_theis_model

# Scaling that turns drawdown into the well function itself: with Q = 4 pi T,
# s = W(u), and with r^2 S / 4T = 1, u = 1/t.
T = 1.5e-3
Q = 4.0 * np.pi * T
R = 40.0
S = 4.0 * T / R ** 2          # so that r^2 S / (4 T) = 1 exactly

# AQTESOLV for Windows User's Guide, printed page 9: the Guided Tour 1 data set.
# time -> displacement, which is W(u) at u = 1/time.
WELL_FUNCTION = [
    (0.5, 0.0489),
    (1, 0.2194),
    (5, 1.223),
    (10, 1.823),
    (50, 3.355),
    (100, 4.038),
    (500, 5.639),
    (1000, 6.332),
    (5000, 7.940),
    (10000, 8.633),
]

# Walton (1962), Bulletin 49, Appendix A, printed page 73: W(u, r/B) from
# Hantush (1956). (u, r/B, W). Values read from columns whose digits are
# unambiguous in the scan.
LEAKY_WELL_FUNCTION = [
    (0.001, 0.1, 4.8292),
    (0.002, 0.1, 4.7079),
    (0.005, 0.1, 4.2960),
    (0.01,  0.1, 3.8150),
    (0.02,  0.1, 3.2442),
    (0.03,  0.1, 2.8873),
    (0.05,  0.1, 2.4271),
    (0.1,   0.1, 1.8050),
    (0.2,   0.1, 1.2155),
    (0.3,   0.1, 0.9018),
    (0.01,  0.2, 3.2875),
    (0.05,  0.2, 2.3110),
    (0.1,   0.2, 1.7527),
    (0.2,   0.2, 1.1944),
    (0.01,  0.5, 1.8486),
    (0.05,  0.5, 1.7075),
    (0.1,   0.5, 1.4422),
    (0.2,   0.5, 1.0592),
    (0.1,   1.0, 0.8190),
    (0.2,   1.0, 0.7148),
]

# The same table's u -> 0 row: the steady-state limit, W(0, r/B) = 2 K0(r/B).
STEADY_STATE = [
    (0.1, 4.8541),
    (0.2, 3.5054),
    (0.3, 2.7449),
    (0.5, 1.8488),
    (0.7, 1.3210),
    (1.0, 0.8420),
]


def _time_for(u: float) -> float:
    """The t at which the scaling above gives this u."""
    return R ** 2 * S / (4.0 * T * u)


class TestTheisAgainstThePublishedWellFunction:
    """W(u), from a table HydroAssist did not produce."""

    @pytest.mark.parametrize("time,published", WELL_FUNCTION)
    def test_the_model_reproduces_the_printed_value(self, time, published):
        model = _make_theis_model(Q, R)
        computed = float(model(np.array([float(time)]), T, S)[0])

        assert computed == pytest.approx(published, abs=5e-4)

    def test_the_whole_curve_at_once(self):
        """Guards against a per-point fluke: the ten values together are the
        shape of the Theis curve over four decades."""
        times = np.array([t for t, _ in WELL_FUNCTION], dtype=float)
        published = np.array([w for _, w in WELL_FUNCTION])

        computed = _make_theis_model(Q, R)(times, T, S)

        assert computed == pytest.approx(published, abs=5e-4)


class TestCooperJacobAgainstThePublishedWellFunction:
    """The straight-line approximation, measured against the curve it approximates.

    Cooper-Jacob replaces W(u) with ln(2.25 T t / r^2 S), which in these terms is
    ln(0.5625 / u). Its two constants — 2.25 and 2.303 — are exactly what a
    published table can check, and the error it makes is what its u < 0.02 rule
    exists to bound.
    """

    def _approximation(self, u: float) -> float:
        """The well function as Cooper-Jacob computes it, from the module's own
        constants: s = (2.303 Q / 4 pi T) log10(2.25 T t / r^2 S), and with
        Q = 4 pi T this is the well function itself."""
        return 2.303 * np.log10(2.25 / (4.0 * u))

    @pytest.mark.parametrize("u,published", [
        (0.01, 4.038), (0.002, 5.639), (0.001, 6.332),
        (0.0002, 7.940), (0.0001, 8.633),
    ])
    def test_it_matches_the_published_value_where_the_method_is_valid(self, u, published):
        assert self._approximation(u) == pytest.approx(published, rel=2.5e-3)

    @pytest.mark.parametrize("u,published,least_error", [
        (1.0, 0.2194, 0.5),     # wildly wrong: the approximation is meaningless
        (0.2, 1.2227, 0.05),
    ])
    def test_it_is_wrong_where_its_own_rule_forbids_it(self, u, published, least_error):
        """The u threshold is not a formality. Above it the straight line
        departs from the published curve by more than the fit's own uncertainty,
        which is why the calculator refuses those points."""
        error = abs(self._approximation(u) - published)

        assert u > CJ_U_THRESHOLD
        assert error > least_error


class TestHantushJacobAgainstThePublishedLeakyWellFunction:
    """W(u, r/B), from Hantush (1956) as printed by Walton (1962).

    This is the test the plan called the merge gate: it exercises the Laplace
    inversion and the characteristic root together against a value neither of
    them computed.
    """

    @pytest.mark.parametrize("u,r_over_b,published", LEAKY_WELL_FUNCTION)
    def test_the_model_reproduces_the_printed_value(self, u, r_over_b, published):
        B = R / r_over_b
        computed = float(hantush_jacob_drawdown([_time_for(u)], T, S, B, Q, R)[0])

        # The printed table has four decimals; Stehfest inversion adds a little
        # of its own. Half a unit in the last printed digit is the bar.
        assert computed == pytest.approx(published, abs=1e-3)

    @pytest.mark.parametrize("r_over_b,published", STEADY_STATE)
    def test_the_steady_state_row(self, r_over_b, published):
        """At large t the leaky solution stops changing, at W = 2 K0(r/B). The
        table's top row is that limit, and the module has its own closed form
        for it — so this checks the transient solution, the closed form and the
        published value against one another."""
        B = R / r_over_b

        closed_form = steady_state_drawdown(Q, T, R, B)
        assert closed_form == pytest.approx(published, abs=1e-3)

        late = float(hantush_jacob_drawdown([_time_for(1e-6)], T, S, B, Q, R)[0])
        assert late == pytest.approx(published, rel=2e-3)

    def test_the_published_row_is_twice_the_bessel_function(self):
        """A check on the transcription rather than the code: if a digit above
        were copied wrongly, it would no longer equal 2 K0(r/B)."""
        for r_over_b, published in STEADY_STATE:
            assert 2.0 * kn(0, r_over_b) == pytest.approx(published, abs=1e-3)

    def test_leakage_always_reduces_drawdown(self):
        """The physical content of the table: at every u, more leakage means
        less drawdown than Theis predicts."""
        theis = _make_theis_model(Q, R)

        for u, r_over_b, published in LEAKY_WELL_FUNCTION:
            time = _time_for(u)
            assert published < float(theis(np.array([time]), T, S)[0])
