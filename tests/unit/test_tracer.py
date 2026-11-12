"""Tracer tests: the two solutions, and everything that keeps them honest.

The radial method rests on an approximation, so it is checked against something
other than itself: `radial_breakthrough` below solves the radial advection-dispersion
equation numerically, and the fit has to recover the dispersivity and the travel
time that went into it.
"""

import io

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import solve_banded
from scipy.special import erfc

from src.agents import method_selection
from src.calculator import registry
from src.calculator.base import CalculationInput
from src.calculator.geometry import AquiferGeometry, TracerGeometry
from src.calculator.methods import tracer as tr
from src.calculator.report import generate_pdf
from src.data.csv_inspector import CONCENTRATION, CSVInspector
from src.data.validator import DataValidator
from src.project.model import Project, Readings, SCHEMA_VERSION, TestSetup
from src.services import analysis, data_intake

NATURAL_SAMPLE = "tests/tracer_natural_sample.csv"
CONVERGENT_SAMPLE = "tests/tracer_convergent_sample.csv"

X, V, ALPHA = 25.0, 0.5 / 86400, 1.5
R, Q, B = 15.0, 500 / 86400, 10.0


def _frame(t, c):
    return pd.DataFrame({"time_s": t, "drawdown_m": c})


def _natural(t, c, **tracer):
    return tr.OgataBanksCalculator().calculate(CalculationInput(
        df=_frame(t, c), r=X, test_type="tracer", tracer=TracerGeometry(**tracer)))


def _radial(t, c, b=B, **tracer):
    return tr.SautyCalculator().calculate(CalculationInput(
        df=_frame(t, c), Q=Q, r=R, test_type="tracer",
        aquifer=AquiferGeometry(b=b), tracer=TracerGeometry(**tracer)))


def _sample(path):
    return data_intake.prepare_upload(pd.read_csv(path)).df


def radial_breakthrough(r0, alpha, nper=2000, tend=2.5):
    """Crank-Nicolson solution of d(rC)/dt = alpha A C_rr + A C_r, for a pulse at
    r0 drawn to a well. Returns (t/t_c, C at the well) with A = 1."""
    A = 1.0
    rw, outer = 0.02 * r0, 1.6 * r0
    dr = min(alpha / 15, r0 / 600)
    n = int((outer - rw) / dr) + 1
    r = np.linspace(rw, outer, n)
    dr = r[1] - r[0]
    t_c = r0 ** 2 / (2 * A)
    dt = t_c / nper

    lo = alpha * A / dr ** 2 - A / (2 * dr)
    up = alpha * A / dr ** 2 + A / (2 * dr)
    di = -2 * alpha * A / dr ** 2
    a, d, c = np.full(n, lo) / r, np.full(n, di) / r, np.full(n, up) / r
    # At the well the gradient vanishes; at the outer edge no mass crosses.
    d[0], c[0], a[0] = di / r[0], -di / r[0], 0.0
    d[-1], a[-1], c[-1] = (di + up * (-2 * dr / alpha)) / r[-1], (lo + up) / r[-1], 0.0

    bands = np.zeros((3, n))
    bands[0, 1:] = -0.5 * dt * c[:-1]
    bands[1, :] = 1 - 0.5 * dt * d
    bands[2, :-1] = -0.5 * dt * a[1:]

    conc = np.exp(-0.5 * ((r - r0) / (3 * dr)) ** 2) / r
    out = [conc[0]]
    for _ in range(int(tend * nper)):
        rhs = conc + 0.5 * dt * d * conc
        rhs[1:] += 0.5 * dt * a[1:] * conc[:-1]
        rhs[:-1] += 0.5 * dt * c[:-1] * conc[1:]
        conc = solve_banded((1, 1), bands, rhs)
        out.append(conc[0])
    return np.arange(len(out)) * dt / t_c, np.array(out)


class TestTheSolutions:

    def test_the_step_response_is_ogata_banks(self):
        """Against the textbook form, written out with its own erfc terms."""
        t = np.array([10.0, 40.0, 50.0, 80.0]) * 86400
        D = ALPHA * V
        direct = 0.5 * (erfc((X - V * t) / (2 * np.sqrt(D * t)))
                        + np.exp(V * X / D) * erfc((X + V * t) / (2 * np.sqrt(D * t))))

        assert tr.natural_continuous(t, V, ALPHA, 1.0, X) == pytest.approx(direct, rel=1e-10)

    def test_a_sharp_front_does_not_overflow(self):
        """exp(vx/D) overflows long before its erfc partner is small."""
        c = tr.natural_continuous(np.array([1e5, 5e6]), V, X / 2000, 1.0, X)

        assert np.all(np.isfinite(c))

    def test_the_step_rises_from_nothing_to_the_plateau(self):
        early, late = tr.natural_continuous(np.array([1e3, 1e9]), V, ALPHA, 7.0, X)

        assert early == pytest.approx(0.0, abs=1e-6)
        assert late == pytest.approx(7.0, rel=1e-6)

    def test_the_pulse_conserves_its_mass_in_space(self):
        """Integrated over distance at a fixed time, the plane source gives back A."""
        x = np.linspace(-200, 400, 40001)
        t = 50 * 86400.0
        D = ALPHA * V
        c = 400.0 / np.sqrt(4 * np.pi * D * t) * np.exp(-(x - V * t) ** 2 / (4 * D * t))

        assert np.trapezoid(c, x) == pytest.approx(400.0, rel=1e-6)
        assert tr.natural_pulse(np.array([t]), V, ALPHA, 400.0, X)[0] == pytest.approx(
            np.interp(X, x, c), rel=1e-4)

    def test_the_radial_pulse_integrates_to_its_amplitude(self):
        t = np.linspace(1.0, 4e7, 2_000_000)
        c = tr.radial_pulse(t, 3e5, 1.0, 1000.0, R)

        assert np.trapezoid(c, t) == pytest.approx(1000.0, rel=1e-3)

    def test_the_radial_mean_arrival_is_later_than_advection(self):
        mean, peclet = tr.radial_moments(1000.0, 3.0, 30.0)

        assert mean == pytest.approx(1200.0)
        assert peclet == pytest.approx(7.5)


class TestAgainstTheRadialEquation:
    """The approximation against a numerical solution it had no part in."""

    @pytest.mark.parametrize("peclet,tolerance", [(10, 0.08), (20, 0.05), (50, 0.05)])
    def test_the_fit_recovers_what_went_in(self, peclet, tolerance):
        r0, alpha = 30.0, 30.0 / peclet
        t_R, c = radial_breakthrough(r0, alpha)

        # Scaled so one unit of dimensionless time is a realistic day.
        t_c = 86400.0
        mask = (t_R > 0.02) & (c > 1e-6 * c.max())
        t, conc = t_R[mask][::20] * t_c, c[mask][::20]
        result = tr.SautyCalculator().calculate(CalculationInput(
            df=_frame(t, conc), Q=Q, r=r0, test_type="tracer",
            tracer=TracerGeometry(conc_unit="relative")))

        assert result.success
        assert result.r_squared > 0.998
        assert result.value("alpha_L") == pytest.approx(alpha, rel=tolerance)
        assert result.value("t_c") == pytest.approx(t_c, rel=tolerance / 2)

    def test_using_the_uniform_flow_peclet_number_would_be_wrong(self):
        """What the 3/4 is for: a uniform-flow reading of this curve overstates the
        spreading. 8/3 is the leading term, so this is taken at a high Peclet number,
        where the terms after it are small."""
        r0, alpha = 30.0, 0.3
        t_R, c = radial_breakthrough(r0, alpha)
        m0 = np.trapezoid(c, t_R)
        mean = np.trapezoid(t_R * c, t_R) / m0
        variance = np.trapezoid((t_R - mean) ** 2 * c, t_R) / m0

        assert mean == pytest.approx(1 + 2 * alpha / r0, rel=0.01)
        assert variance * r0 / alpha == pytest.approx(8 / 3, rel=0.10)


class TestOgataBanks:

    def test_it_recovers_a_noiseless_pulse(self):
        t = np.arange(1, 41) * 4 * 86400.0
        result = _natural(t, tr.natural_pulse(t, V, ALPHA, 400.0, X))

        assert result.success
        assert result.value("v") == pytest.approx(V, rel=1e-4)
        assert result.value("alpha_L") == pytest.approx(ALPHA, rel=1e-4)
        assert result.value("A_over_n") == pytest.approx(400.0, rel=1e-4)

    def test_it_recovers_a_noiseless_step(self):
        t = np.arange(1, 41) * 4 * 86400.0
        result = _natural(t, tr.natural_continuous(t, V, ALPHA, 80.0, X),
                          injection="continuous")

        assert result.value("v") == pytest.approx(V, rel=1e-4)
        assert result.value("alpha_L") == pytest.approx(ALPHA, rel=1e-4)
        assert result.value("C0") == pytest.approx(80.0, rel=1e-4)

    def test_a_known_plateau_is_held_rather_than_fitted(self):
        t = np.arange(1, 41) * 4 * 86400.0
        result = _natural(t, tr.natural_continuous(t, V, ALPHA, 80.0, X),
                          injection="continuous", C0=80.0)

        assert result.get("C0").fixed
        assert result.gof.n_params == 2

    def test_the_background_is_not_mistaken_for_tracer(self):
        t = np.arange(1, 41) * 4 * 86400.0
        c = 3.0 + tr.natural_pulse(t, V, ALPHA, 400.0, X)
        result = _natural(t, c, background=3.0)

        assert result.value("alpha_L") == pytest.approx(ALPHA, rel=1e-3)
        assert result.drawdown_fitted.min() >= 3.0

    def test_it_reads_the_bundled_sample(self):
        df = _sample(NATURAL_SAMPLE)
        result = _natural(df.time_s.values, df.drawdown_m.values)

        assert result.r_squared > 0.99
        assert result.value("v") * 86400 == pytest.approx(0.5, rel=0.05)
        assert result.value("alpha_L") == pytest.approx(ALPHA, rel=0.2)
        assert result.get("v").ci[0] < result.value("v") < result.get("v").ci[1]

    def test_porosity_comes_from_darcy_when_k_and_the_gradient_are_given(self):
        df = _sample(NATURAL_SAMPLE)
        K, i = 5.0 / 86400, 0.02
        result = _natural(df.time_s.values, df.drawdown_m.values, K=K, gradient=i)

        n_e = result.get("n_e")
        assert n_e.derived
        assert n_e.value == pytest.approx(K * i / result.value("v"))

    def test_without_them_it_is_not_reported(self):
        df = _sample(NATURAL_SAMPLE)
        result = _natural(df.time_s.values, df.drawdown_m.values)

        assert result.get("n_e") is None

    def test_it_needs_no_pumping_rate(self):
        df = _sample(NATURAL_SAMPLE)

        assert _natural(df.time_s.values, df.drawdown_m.values).success


class TestSauty:

    def test_it_recovers_a_noiseless_pulse(self):
        t = np.arange(1, 41) * 6 * 3600.0
        result = _radial(t, tr.radial_pulse(t, 2.4e5, 1.0, 5e5, R))

        assert result.value("t_c") == pytest.approx(2.4e5, rel=1e-4)
        assert result.value("alpha_L") == pytest.approx(1.0, rel=1e-4)
        assert result.value("M_over_Q") == pytest.approx(5e5, rel=1e-4)

    def test_it_recovers_a_noiseless_step(self):
        t = np.arange(1, 41) * 6 * 3600.0
        result = _radial(t, tr.radial_continuous(t, 2.4e5, 1.0, 12.0, R),
                         injection="continuous")

        assert result.value("t_c") == pytest.approx(2.4e5, rel=1e-4)
        assert result.value("C0") == pytest.approx(12.0, rel=1e-4)

    def test_porosity_and_recovery_from_the_bundled_sample(self):
        df = _sample(CONVERGENT_SAMPLE)
        result = _radial(df.time_s.values, df.drawdown_m.values, M_injected=5.0)

        assert result.r_squared > 0.99
        assert result.value("n_e") == pytest.approx(0.2, rel=0.05)
        assert result.value("alpha_L") == pytest.approx(1.0, rel=0.2)
        assert result.value("M_recovered") == pytest.approx(4.0, rel=0.05)
        assert result.value("recovery") == pytest.approx(0.8, rel=0.05)

    def test_a_microgram_reading_is_not_a_milligram_of_mass(self):
        df = _sample(CONVERGENT_SAMPLE)
        mg = _radial(df.time_s.values, df.drawdown_m.values, conc_unit="mg/L")
        ug = _radial(df.time_s.values, df.drawdown_m.values, conc_unit="µg/L")

        assert ug.value("M_recovered") == pytest.approx(mg.value("M_recovered") / 1000)

    def test_relative_concentration_has_no_mass_to_report(self):
        df = _sample(CONVERGENT_SAMPLE)
        result = _radial(df.time_s.values, df.drawdown_m.values, conc_unit="relative",
                         M_injected=5.0)

        assert result.get("M_recovered") is None and result.get("recovery") is None

    def test_it_needs_the_pumping_rate(self):
        df = _sample(CONVERGENT_SAMPLE)
        result = tr.SautyCalculator().calculate(CalculationInput(
            df=df, r=R, test_type="tracer", tracer=TracerGeometry()))

        assert not result.success and "Q" in result.error_message


class TestTheChecksAndTheNotes:

    def _notes(self, result):
        return {f.id for f in result.findings}

    def test_negative_readings_are_refused(self):
        t = np.arange(1, 21) * 3600.0
        result = _natural(t, np.linspace(-1, 3, 20))

        assert not result.success and "Negative" in result.error_message

    def test_a_flat_record_has_no_breakthrough(self):
        result = _natural(np.arange(1, 21) * 3600.0, np.full(20, 2.0), background=2.0)

        assert not result.success and "no breakthrough" in result.error_message

    def test_too_few_points_are_refused(self):
        result = _natural(np.arange(1, 6) * 3600.0, np.arange(1.0, 6.0))

        assert not result.success and "Insufficient" in result.error_message

    def test_a_peak_that_was_never_seen_is_called_out(self):
        t = np.arange(1, 31) * 86400.0
        result = _natural(t, tr.natural_pulse(t, V, ALPHA, 400.0, X))

        assert "peak_not_captured" in self._notes(result)

    def test_a_record_stopped_in_the_tail_is_called_out(self):
        t = np.arange(1, 21) * 3.25 * 86400.0
        result = _natural(t, tr.natural_pulse(t, V, ALPHA, 400.0, X))

        assert "tail_truncated" in self._notes(result)

    def test_a_complete_record_earns_neither(self):
        df = _sample(NATURAL_SAMPLE)
        notes = self._notes(_natural(df.time_s.values, df.drawdown_m.values))

        assert not notes & {"peak_not_captured", "tail_truncated"}

    def test_a_plateau_not_reached_is_called_out(self):
        t = np.arange(1, 16) * 4 * 86400.0
        result = _natural(t, tr.natural_continuous(t, V, ALPHA, 80.0, X),
                          injection="continuous")

        assert "plateau_not_reached" in self._notes(result)

    def test_a_low_peclet_number_is_called_out_for_the_radial_method(self):
        t = np.arange(1, 41) * 6 * 3600.0
        result = _radial(t, tr.radial_pulse(t, 2.4e5, 3.0, 5e5, R))

        assert "radial_peclet_low" in self._notes(result)

    def test_more_mass_than_was_injected_is_called_out(self):
        df = _sample(CONVERGENT_SAMPLE)
        result = _radial(df.time_s.values, df.drawdown_m.values, M_injected=1.0)

        assert "recovery_above_one" in self._notes(result)

    def test_an_implausible_porosity_is_called_out(self):
        df = _sample(CONVERGENT_SAMPLE)
        result = _radial(df.time_s.values, df.drawdown_m.values, b=1.0)

        assert "porosity_implausible" in self._notes(result)


class TestRegistry:

    @pytest.mark.parametrize("text,expected", [
        ("Ogata-Banks", "ogata_banks_1961"),
        ("use ogata banks", "ogata_banks_1961"),
        ("Sauty", "sauty_1980"),
        ("a radial convergent test", "sauty_1980"),
        ("natural gradient tracer test", "ogata_banks_1961"),
    ])
    def test_the_names_resolve(self, text, expected):
        assert registry.canonical_id(text) == expected

    def test_a_question_about_the_gradient_is_not_a_tracer_method(self):
        assert registry.canonical_id("what is the hydraulic gradient here?") is None

    def test_they_are_not_offered_for_an_aquifer_type(self):
        """Tied to no confining condition, so the chat's aquifer rules stay as they
        were: asking about a leaky aquifer must not produce a tracer method."""
        assert registry.for_aquifer(registry.LEAKY)[0].id == "hantush_jacob_1955"
        assert all(registry.TRACER not in s.test_types
                   for a in (registry.CONFINED, registry.LEAKY,
                             registry.UNCONFINED, registry.FRACTURED)
                   for s in registry.for_aquifer(a))

    def test_only_they_are_offered_for_a_tracer_test(self):
        assert [s.id for s in registry.for_test_type(registry.TRACER)] == [
            "ogata_banks_1961", "sauty_1980"]


class TestChoosingFromThePlainWords:

    @pytest.mark.parametrize("message,expected", [
        ("I injected a tracer and sampled the downgradient well",
         "Ogata-Banks (1961)"),
        ("We ran a bromide breakthrough curve under the natural gradient",
         "Ogata-Banks (1961)"),
        ("I injected fluorescein 20 m from the pumped well", "Sauty (1980)"),
        ("How do I analyse a convergent tracer test?", "Sauty (1980)"),
    ])
    def test_the_arrangement_picks_the_method(self, message, expected):
        assert method_selection.suggest_method(user_message=message)["method"] == expected

    def test_a_pumping_question_is_untouched(self):
        assert not method_selection.describes_tracer("my pumping test in a confined aquifer")

    def test_a_tracer_result_is_summarised_without_t_or_s(self):
        df = _sample(NATURAL_SAMPLE)
        text = method_selection.summarise_result(
            _natural(df.time_s.values, df.drawdown_m.values).to_state_dict())

        assert "m/day" in text and "mg/L" in text
        assert "**T**" not in text and "**S**" not in text and " mm" not in text


class TestReadingTheFile:

    def _inspect(self, *columns, tracer=False):
        data = {name: np.arange(1.0, 13.0) for name in columns}
        return CSVInspector().inspect(pd.DataFrame(data), tracer=tracer)

    @pytest.mark.parametrize("name", [
        "concentration", "Concentration (mg/L)", "conc_ppb", "C", "c_mg_l", "C/C0",
        "bromide", "tracer_ug_L", "chloride_mg/L",
    ])
    def test_a_concentration_column_is_recognised(self, name):
        found = self._inspect("time_min", name)

        assert found.drawdown_cols == [name]
        assert found.drawdown_unit == CONCENTRATION

    @pytest.mark.parametrize("name", ["drawdown_m", "s", "water_level_ft", "dd"])
    def test_a_drawdown_column_is_not(self, name):
        assert self._inspect("time_min", name).drawdown_unit != CONCENTRATION

    @pytest.mark.parametrize("extra", ["temp_c", "conductivity_uS", "barometric"])
    def test_a_logger_channel_does_not_turn_it_into_a_tracer_test(self, extra):
        found = self._inspect("time_min", "drawdown_m", extra)

        assert found.drawdown_cols[0] == "drawdown_m"
        assert found.drawdown_unit != CONCENTRATION

    def test_a_chloride_channel_beside_a_water_level_stays_a_pumping_test(self):
        assert self._inspect("time_min", "wl_m", "chloride").drawdown_unit != CONCENTRATION

    def test_a_large_concentration_is_not_converted_from_feet(self):
        """The drawdown fallback reads any value above 150 as feet."""
        raw = pd.DataFrame({"time_h": np.arange(1.0, 13.0),
                            "conc": np.arange(1.0, 13.0) * 100})
        prepared = data_intake.prepare_upload(raw)

        assert prepared.df.drawdown_m.max() == pytest.approx(1200.0)

    def test_on_a_tracer_test_an_unnamed_column_is_concentration_and_says_so(self):
        found = self._inspect("time_min", "value", tracer=True)

        assert found.drawdown_cols == ["value"]
        assert found.drawdown_unit == CONCENTRATION
        assert any("concentration" in issue for issue in found.issues)

    def test_the_llm_is_not_asked_about_drawdown_units_for_a_tracer_file(self):
        class Explodes:
            def invoke(self, prompt):
                raise AssertionError("the LLM should not be called")

        raw = pd.DataFrame({"reading": np.arange(1.0, 13.0), "conc": np.arange(1.0, 13.0)})
        CSVInspector(llm=Explodes()).inspect(raw)

    def test_the_upload_decides_the_test_type(self):
        prepared = data_intake.prepare_upload(pd.read_csv(NATURAL_SAMPLE))

        assert prepared.test_type == registry.TRACER
        assert prepared.quantity == "concentration"

    def test_no_flow_regime_is_read_from_a_breakthrough_curve(self):
        prepared = data_intake.prepare_upload(pd.read_csv(NATURAL_SAMPLE))

        assert prepared.diagnosis is None
        assert prepared.consistent_methods == []

    def test_the_drawdown_checks_do_not_fire_on_a_breakthrough_curve(self):
        df = _sample(NATURAL_SAMPLE)
        validation = DataValidator().validate(df, test_type=registry.TRACER)

        assert validation.is_usable
        assert not any("drawdown" in w.lower() or "log cycle" in w
                       for w in validation.warnings + validation.info)
        assert set(validation.suitable_methods) == {"Ogata-Banks (1961)", "Sauty (1980)"}


class TestTheWrongKindOfReadings:

    def test_theis_refuses_concentrations(self):
        readings = Readings.from_frame(_sample(NATURAL_SAMPLE), quantity="concentration")

        with pytest.raises(ValueError, match="fitted to drawdown"):
            analysis.check_readings_suit(readings, "theis_1935")

    def test_a_tracer_method_refuses_drawdown(self):
        readings = Readings.from_frame(_frame([1.0, 2.0], [0.1, 0.2]))

        with pytest.raises(ValueError, match="fitted to concentrations"):
            analysis.check_readings_suit(readings, "sauty_1980")

    def test_the_matching_kinds_pass(self):
        conc = Readings.from_frame(_sample(NATURAL_SAMPLE), quantity="concentration")
        analysis.check_readings_suit(conc, "ogata_banks_1961")
        analysis.check_readings_suit(Readings.from_frame(_frame([1.0], [0.1])), "theis_1935")


class TestTheProjectFile:

    def _project(self):
        project = Project()
        test = project.add_test()
        data_intake.store_readings(test, _sample(CONVERGENT_SAMPLE),
                                   quantity="concentration")
        test.setup = TestSetup(
            Q_m3s=Q, r_m=R, test_type="tracer", tracer_injection="pulse",
            tracer_conc_unit="µg/L", tracer_background=0.2, tracer_M_kg=5.0,
            tracer_b_m=B)
        return project, test

    def test_it_all_survives_the_round_trip(self):
        project, test = self._project()
        reopened = Project.from_json(project.to_json()).tests[0]

        assert reopened.readings.quantity == "concentration"
        assert reopened.readings.fingerprint == test.readings.fingerprint
        assert vars(reopened.setup) == vars(test.setup)

    def test_an_older_file_opens_as_drawdown(self):
        project, _ = self._project()
        data = project.to_dict()
        data["schema_version"] = 2
        del data["tests"][0]["readings"]["quantity"]
        for name in list(data["tests"][0]["setup"]):
            if name.startswith("tracer_"):
                del data["tests"][0]["setup"][name]

        reopened = Project.from_dict(data).tests[0]

        assert reopened.readings.quantity == "drawdown"
        assert reopened.setup.tracer_injection == "pulse"

    def test_a_drawdown_fingerprint_has_not_moved(self):
        """Saved analyses compare against it, so a change would mark every analysis
        in every existing project stale."""
        import hashlib
        import json

        readings = Readings(time_s=[1.0, 2.0], drawdown_m=[0.1, 0.2])
        expected = hashlib.sha256(
            json.dumps([[1.0, 2.0], [0.1, 0.2], None]).encode()).hexdigest()[:16]

        assert readings.fingerprint == expected

    def test_the_format_version_went_up(self):
        assert SCHEMA_VERSION == 3

    def test_a_saved_analysis_restores_with_its_tracer_inputs(self):
        project, test = self._project()
        df = test.readings.to_frame()
        setup = test.setup
        result = analysis.run_analysis(df, "sauty_1980", setup)
        analysis.save_analysis(test, result, analysis.run_record(setup, "sauty_1980", result))

        restored, record = analysis.restore_last_analysis(
            Project.from_json(project.to_json()).tests[0])

        assert restored.value("n_e") == pytest.approx(result.value("n_e"))
        assert restored.curves["tracer"]["unit"] == "µg/L"
        assert record["tracer_b_m"] == B

    def test_a_run_is_not_logged_as_zero_transmissivity(self):
        project, test = self._project()
        result = analysis.run_analysis(test.readings.to_frame(), "sauty_1980", test.setup)
        run = test.record_run(result, "Sauty (1980)", "sauty_1980")

        assert run.T_day is None and run.S is None

    def test_the_comparison_shows_what_was_estimated_instead(self):
        project, test = self._project()
        setup = test.setup
        result = analysis.run_analysis(test.readings.to_frame(), "sauty_1980", setup)
        analysis.save_analysis(test, result, analysis.run_record(setup, "sauty_1980", result))

        row = test.comparison()[0]

        assert row["T_m2_per_day"] is None and row["S"] is None
        assert "α_L" in row["estimates"] and "n_e" in row["estimates"]


class TestTheReport:

    def _text(self, method, path, **setup):
        from pypdf import PdfReader

        df = _sample(path)
        setup = TestSetup(test_type="tracer", **setup)
        result = analysis.run_analysis(df, method, setup)
        record = analysis.run_record(setup, method, result)
        pdf, name = analysis.report_pdf(result, record, df)
        assert name.endswith("_report.pdf")
        return " ".join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)

    def test_it_reads_as_a_tracer_report(self):
        text = self._text("sauty_1980", CONVERGENT_SAMPLE, Q_m3s=Q, r_m=R, tracer_b_m=B)

        assert "Tracer Test Analysis" in text and "Concentration (mg/L)" in text
        assert "Effective porosity" in text and "Travel distance" in text
        assert "Pumping Test" not in text and "Drawdown" not in text
        assert "Transmissivity" not in text and " mm" not in text

    def test_a_natural_gradient_report_mentions_no_pumping_rate(self):
        text = self._text("ogata_banks_1961", NATURAL_SAMPLE, r_m=X)

        assert "Pumping rate" not in text

    def test_generate_pdf_works_without_a_rate(self):
        df = _sample(NATURAL_SAMPLE)
        result = _natural(df.time_s.values, df.drawdown_m.values)

        assert generate_pdf(result, 0.0, X, "Ogata-Banks (1961)", df).startswith(b"%PDF-")
