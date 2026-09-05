"""Tests for the geometry dataclasses and their reconciliation with CalculationInput.

Two things are being pinned here.

First, the invariants. Each geometry group rejects the physically impossible at
construction rather than letting it reach a solver and emerge as a plausible but
wrong number: an aquifer bottom above its top, Kz greater than Kr, a screen whose
bottom is above its top, a pumping schedule that does not start at t = 0.

Second, backward compatibility. `CalculationInput` gained eight fields, and the
104 existing calculator tests construct it the old way. The alias reconciliation
has to leave `Q`, `r`, `r_w`, `r_c` and every error message exactly as they were.
"""

import numpy as np
import pandas as pd
import pytest

from src.calculator import registry
from src.calculator.base import CalculationInput
from src.calculator.geometry import (
    CONFINED, AquiferGeometry, AquitardGeometry, FractureGeometry,
    ObservationGeometry, PumpingSchedule, RateStep, SlugTestGeometry,
    WellFieldMember, WellGeometry,
)


def _df(n=30):
    t = np.logspace(1, 4, n)
    return pd.DataFrame({"time_s": t, "drawdown_m": 0.15 * np.log10(t)})


# ── Vocabulary ────────────────────────────────────────────────────────────────

class TestVocabularyMatchesTheRegistry:
    """The geometry layer and the method table must agree on their terms."""

    def test_aquifer_types_agree(self):
        from src.calculator import geometry as g
        assert {g.CONFINED, g.LEAKY, g.UNCONFINED, g.FRACTURED} == set(
            registry.AQUIFER_TYPES)

    def test_test_types_agree(self):
        from src.calculator import geometry as g
        assert {g.CONSTANT_RATE, g.RECOVERY, g.STEP_DRAWDOWN, g.SLUG,
                g.CONSTANT_HEAD} == set(registry.TEST_TYPES)


# ── Individual groups ─────────────────────────────────────────────────────────

class TestAquiferGeometry:

    def test_empty_is_valid(self):
        """Everything is optional; the three confined methods need none of it."""
        assert AquiferGeometry().b is None

    def test_rejects_non_positive_thickness(self):
        with pytest.raises(ValueError, match="thickness b must be positive"):
            AquiferGeometry(b=0.0)

    @pytest.mark.parametrize("ratio", [0.0, -0.1, 1.5])
    def test_rejects_impossible_anisotropy(self, ratio):
        """Kz > Kr would mean water crosses bedding more easily than it follows
        it, which layered sediments do not do."""
        with pytest.raises(ValueError, match="Kz/Kr"):
            AquiferGeometry(Kz_over_Kr=ratio)

    def test_accepts_isotropic_and_typical_anisotropy(self):
        assert AquiferGeometry(Kz_over_Kr=1.0).Kz_over_Kr == 1.0
        assert AquiferGeometry(Kz_over_Kr=0.1).Kz_over_Kr == 0.1

    def test_rejects_inverted_depths(self):
        with pytest.raises(ValueError, match="must be below its top"):
            AquiferGeometry(top_m=50.0, bottom_m=20.0)

    def test_defaults_to_confined(self):
        assert AquiferGeometry().aquifer_type == CONFINED


class TestWellGeometry:

    def test_defaults_match_the_legacy_scalars(self):
        w = WellGeometry()
        assert (w.r_w, w.r_c) == (0.10, 0.10)

    @pytest.mark.parametrize("kw,msg", [
        ({"r_w": 0.0}, "r_w must be positive"),
        ({"r_c": -1.0}, "r_c must be positive"),
    ])
    def test_rejects_non_positive_radii(self, kw, msg):
        with pytest.raises(ValueError, match=msg):
            WellGeometry(**kw)

    def test_rejects_inverted_screen(self):
        with pytest.raises(ValueError, match="must be below its top"):
            WellGeometry(screen_top_m=30.0, screen_bottom_m=10.0)

    def test_screen_length_is_derived(self):
        w = WellGeometry(screen_top_m=10.0, screen_bottom_m=25.0)
        assert w.screen_length == pytest.approx(15.0)

    def test_screen_length_is_none_when_undefined(self):
        assert WellGeometry().screen_length is None


class TestObservationGeometry:

    def test_requires_a_positive_distance(self):
        with pytest.raises(ValueError, match="r must be positive"):
            ObservationGeometry(r=0.0)

    def test_carries_an_optional_well_id(self):
        assert ObservationGeometry(r=50.0, well_id="MW-1").well_id == "MW-1"


class TestAquitardGeometry:

    def test_leakance_is_derived(self):
        a = AquitardGeometry(b_prime=5.0, K_prime=1e-8)
        assert a.leakance == pytest.approx(2e-9)

    def test_leakance_is_none_when_incomplete(self):
        assert AquitardGeometry(b_prime=5.0).leakance is None

    def test_rejects_an_unknown_position(self):
        with pytest.raises(ValueError, match="'upper' or 'lower'"):
            AquitardGeometry(position="sideways")

    @pytest.mark.parametrize("field", ["b_prime", "K_prime", "Ss_prime"])
    def test_rejects_non_positive_properties(self, field):
        with pytest.raises(ValueError, match="must be positive"):
            AquitardGeometry(**{field: -1.0})


class TestFractureGeometry:

    def test_defaults_to_slab_blocks(self):
        assert FractureGeometry().block_shape == "slab"

    def test_rejects_an_unknown_block_shape(self):
        with pytest.raises(ValueError, match="'slab' or 'sphere'"):
            FractureGeometry(block_shape="cube")

    def test_rejects_a_non_positive_block_width(self):
        with pytest.raises(ValueError, match="half-width must be positive"):
            FractureGeometry(block_half_width=0.0)

    def test_rejects_an_empty_series(self):
        with pytest.raises(ValueError, match="at least 1"):
            FractureGeometry(n_block_terms=0)


class TestSlugTestGeometry:

    def test_rejects_zero_displacement(self):
        """With H0 = 0 there is no test."""
        with pytest.raises(ValueError, match="cannot be zero"):
            SlugTestGeometry(H0=0.0, r_casing=0.05, r_screen=0.1)

    def test_accepts_a_falling_head_test(self):
        """A negative H0 is a rising-head test — legitimate."""
        assert SlugTestGeometry(H0=-0.5, r_casing=0.05, r_screen=0.1).H0 == -0.5

    def test_rejects_non_positive_radii(self):
        with pytest.raises(ValueError, match="r_casing must be positive"):
            SlugTestGeometry(H0=0.5, r_casing=0.0, r_screen=0.1)


# ── Pumping schedule ──────────────────────────────────────────────────────────

class TestPumpingSchedule:

    def test_single_step_is_constant_rate(self):
        s = PumpingSchedule(steps=(RateStep(0.0, 0.006),))
        assert s.is_constant
        assert s.initial_rate == pytest.approx(0.006)

    def test_requires_at_least_one_step(self):
        with pytest.raises(ValueError, match="at least one rate step"):
            PumpingSchedule(steps=())

    def test_must_start_at_zero(self):
        with pytest.raises(ValueError, match="must start at t = 0"):
            PumpingSchedule(steps=(RateStep(100.0, 0.006),))

    def test_rejects_duplicate_start_times(self):
        with pytest.raises(ValueError, match="same time"):
            PumpingSchedule(steps=(RateStep(0.0, 0.006), RateStep(0.0, 0.01)))

    def test_steps_are_sorted(self):
        s = PumpingSchedule(steps=(RateStep(1000.0, 0.01), RateStep(0.0, 0.006)))
        assert [x.t_start_s for x in s.steps] == [0.0, 1000.0]

    def test_rate_at_returns_the_step_in_force(self):
        s = PumpingSchedule(steps=(RateStep(0.0, 0.006), RateStep(1000.0, 0.012)))

        assert s.rate_at(0.0) == pytest.approx(0.006)
        assert s.rate_at(999.0) == pytest.approx(0.006)
        assert s.rate_at(1000.0) == pytest.approx(0.012)
        assert s.rate_at(5000.0) == pytest.approx(0.012)

    def test_rate_changes_are_the_superposition_increments(self):
        """Variable-rate drawdown sums delta_Q_i * s_unit(t - t_i)."""
        s = PumpingSchedule(steps=(RateStep(0.0, 0.006),
                                   RateStep(1000.0, 0.012),
                                   RateStep(2000.0, 0.0)))
        changes = s.rate_changes()

        assert changes[0] == (0.0, pytest.approx(0.006))
        assert changes[1] == (1000.0, pytest.approx(0.006))
        assert changes[2] == (2000.0, pytest.approx(-0.012))

    def test_increments_sum_to_the_final_rate(self):
        s = PumpingSchedule(steps=(RateStep(0.0, 0.006), RateStep(1000.0, 0.012)))
        assert sum(d for _, d in s.rate_changes()) == pytest.approx(0.012)

    def test_recovery_is_a_final_zero_step(self):
        s = PumpingSchedule(steps=(RateStep(0.0, 0.006), RateStep(3600.0, 0.0)))
        assert s.rate_at(4000.0) == 0.0


class TestWellField:

    def test_distance_is_euclidean(self):
        w = WellFieldMember("PW-1", 0.0, 0.0,
                            PumpingSchedule(steps=(RateStep(0.0, 0.006),)))
        assert w.distance_to(3.0, 4.0) == pytest.approx(5.0)


# ── Reconciliation with CalculationInput ──────────────────────────────────────

class TestBackwardCompatibility:
    """The 104 shipped calculator tests construct inputs the old way."""

    def test_legacy_construction_is_unchanged(self):
        inp = CalculationInput(df=_df(), Q=500 / 86400, r=50.0)

        assert inp.Q == pytest.approx(500 / 86400)
        assert inp.r == 50.0
        assert (inp.r_w, inp.r_c) == (0.10, 0.10)

    def test_papadopulos_style_construction_is_unchanged(self):
        inp = CalculationInput(df=_df(), Q=300 / 86400, r=0.30,
                               r_w=0.30, r_c=0.20)
        assert (inp.r_w, inp.r_c) == (0.30, 0.20)

    @pytest.mark.parametrize("kw,msg", [
        ({"Q": 500.0, "r": 50.0}, "unrealistically large"),
        ({"Q": -1.0, "r": 50.0}, "Q must be positive"),
        ({"Q": 0.006, "r": 0.0}, "r must be positive"),
        ({"Q": 0.006, "r": 50.0, "r_w": 0.0}, "r_w must be positive"),
        ({"Q": 0.006, "r": 50.0, "r_c": -1.0}, "r_c must be positive"),
    ])
    def test_every_legacy_error_message_survives(self, kw, msg):
        with pytest.raises(ValueError, match=msg):
            CalculationInput(df=_df(), **kw)

    def test_get_series_is_unchanged(self):
        inp = CalculationInput(df=_df(), Q=0.006, r=50.0)
        t, s = inp.get_series()
        assert len(t) == 30 and len(s) == 30


class TestAliasReconciliation:

    def test_scalars_populate_the_groups(self):
        inp = CalculationInput(df=_df(), Q=0.006, r=50.0, r_w=0.3, r_c=0.2)

        assert inp.well.r_w == 0.3
        assert inp.well.r_c == 0.2
        assert inp.observation.r == 50.0

    def test_groups_populate_the_scalars(self):
        inp = CalculationInput(df=_df(), Q=0.006,
                               observation=ObservationGeometry(r=75.0))
        assert inp.r == 75.0

    def test_supplying_both_consistently_is_fine(self):
        inp = CalculationInput(df=_df(), Q=0.006, r=50.0, r_w=0.3, r_c=0.2,
                               well=WellGeometry(r_w=0.3, r_c=0.2))
        assert inp.well.r_w == 0.3

    def test_conflicting_geometry_is_rejected_loudly(self):
        """Silently preferring one would make the wrong curve plausible."""
        with pytest.raises(ValueError, match="Conflicting well geometry"):
            CalculationInput(df=_df(), Q=0.006, r=50.0, r_w=0.3, r_c=0.2,
                             well=WellGeometry(r_w=0.9, r_c=0.9))

    def test_aquifer_group_is_always_present(self):
        assert CalculationInput(df=_df(), Q=0.006, r=50.0).aquifer is not None

    def test_a_constant_rate_schedule_is_synthesised(self):
        inp = CalculationInput(df=_df(), Q=0.006, r=50.0)

        assert inp.schedule.is_constant
        assert inp.schedule.rate_at(1000.0) == pytest.approx(0.006)

    def test_an_explicit_schedule_is_preserved(self):
        schedule = PumpingSchedule(steps=(RateStep(0.0, 0.006),
                                          RateStep(3600.0, 0.0)))
        inp = CalculationInput(df=_df(), Q=0.006, r=50.0, schedule=schedule)
        assert inp.schedule.rate_at(4000.0) == 0.0

    def test_well_id_flows_into_the_observation_group(self):
        df = _df()
        df["well_id"] = "MW-1"
        inp = CalculationInput(df=df, Q=0.006, r=50.0, well_id="MW-1")
        assert inp.observation.well_id == "MW-1"


class TestSlugTestInputs:
    """A slug test has no pumping rate, which the guards must accommodate."""

    def test_slug_test_needs_no_rate_or_distance(self):
        inp = CalculationInput(
            df=_df(), test_type="slug",
            slug=SlugTestGeometry(H0=0.5, r_casing=0.05, r_screen=0.1),
        )
        assert inp.Q is None
        assert inp.slug.H0 == 0.5

    def test_a_pumping_test_still_requires_a_rate(self):
        with pytest.raises(ValueError, match="Q is required"):
            CalculationInput(df=_df(), r=50.0)

    def test_slug_geometry_is_reachable(self):
        inp = CalculationInput(
            df=_df(), test_type="slug",
            slug=SlugTestGeometry(H0=0.5, r_casing=0.05, r_screen=0.1,
                                  screen_length=3.0),
        )
        assert inp.slug.screen_length == 3.0


class TestRicherGeometry:

    def test_aquitards_can_be_attached(self):
        inp = CalculationInput(
            df=_df(), Q=0.006, r=50.0,
            aquitards=(AquitardGeometry(b_prime=5.0, K_prime=1e-8),),
        )
        assert inp.aquitards[0].leakance == pytest.approx(2e-9)

    def test_unconfined_geometry_can_be_attached(self):
        inp = CalculationInput(
            df=_df(), Q=0.006, r=50.0,
            aquifer=AquiferGeometry(b=20.0, Kz_over_Kr=0.1,
                                    aquifer_type=registry.UNCONFINED),
        )
        assert inp.aquifer.b == 20.0

    def test_options_carry_per_method_overrides(self):
        inp = CalculationInput(df=_df(), Q=0.006, r=50.0,
                               options={"u_threshold": 0.05})
        assert inp.options["u_threshold"] == 0.05

    def test_geometry_paths_resolve_for_require_geometry(self):
        """The dotted paths in a registry `requires` tuple must actually resolve."""
        from src.calculator.validation import resolve_path

        inp = CalculationInput(
            df=_df(), Q=0.006, r=50.0, r_w=0.3, r_c=0.2,
            aquifer=AquiferGeometry(b=20.0),
            aquitards=(AquitardGeometry(b_prime=5.0),),
        )
        assert resolve_path(inp, "well.r_w") == 0.3
        assert resolve_path(inp, "aquifer.b") == 20.0
        assert resolve_path(inp, "aquitards[0].b_prime") == 5.0
        assert resolve_path(inp, "fracture.block_half_width") is None

    def test_papadopulos_cooper_requirements_are_satisfied_by_its_call_shape(self):
        """The registry says PC requires well.r_w and well.r_c; the way app.py
        constructs its input must actually satisfy that."""
        from src.calculator.validation import require_geometry

        spec = registry.get("papadopulos_cooper_1967")
        inp = CalculationInput(df=_df(), Q=300 / 86400, r=0.30,
                               r_w=0.30, r_c=0.20)
        assert require_geometry(inp, spec.requires, spec.label) is None
