"""Unit tests for method selection — the chat → calculator decision logic.

This module is deliberately LLM-free so the rule that produced a suggestion is
always traceable. These tests pin the rules to the source papers:

  * Theis (1935) assumption (5) — infinitesimal well diameter — is what makes a
    large-diameter well a Papadopulos-Cooper case.
  * Cooper-Jacob (1946) p. 3 — the straight-line method needs u < 0.02, i.e.
    late-time data.
  * Only confined aquifers have an implemented calculator.
"""

import pytest

from src.agents.method_selection import (
    THEIS, COOPER_JACOB, PAPADOPULOS_COOPER, SUPPORTED_METHODS,
    LARGE_WELL_RADIUS_M,
    canonical_method, extract_well_radius_m, suggest_method,
    format_suggestion, summarise_result,
)


# ── Canonical method names ────────────────────────────────────────────────────

class TestCanonicalMethod:

    @pytest.mark.parametrize("raw,expected", [
        ("Theis", THEIS),
        ("theis", THEIS),
        ("Theis (1935)", THEIS),
        ("the theis method", THEIS),
        ("Cooper-Jacob", COOPER_JACOB),
        ("cooper jacob", COOPER_JACOB),
        ("Cooper_Jacob (1946)", COOPER_JACOB),
        ("jacob straight line", COOPER_JACOB),
        ("Papadopulos-Cooper", PAPADOPULOS_COOPER),
        ("papadopulos cooper 1967", PAPADOPULOS_COOPER),
    ])
    def test_aliases_map_to_canonical_labels(self, raw, expected):
        assert canonical_method(raw) == expected

    def test_papadopulos_wins_over_cooper_jacob(self):
        """Both names contain 'Cooper' — the more specific one must win."""
        assert canonical_method("Papadopulos-Cooper (1967)") == PAPADOPULOS_COOPER

    @pytest.mark.parametrize("raw", [None, "", "xyz", "Neuman (1972)",
                                     "Boulton", "Moench"])
    def test_unknown_or_unimplemented_returns_none(self, raw):
        """Real-but-unimplemented methods must not be silently coerced."""
        assert canonical_method(raw) is None

    def test_hantush_jacob_no_longer_collides_with_cooper_jacob(self):
        """Both names contain 'Jacob'. Previously "Hantush-Jacob" resolved to
        Cooper-Jacob, which was recorded as a known limitation; implementing the
        method resolved it, because the longer alias now wins."""
        from src.agents.method_selection import HANTUSH_JACOB

        assert canonical_method("Hantush-Jacob") == HANTUSH_JACOB
        assert canonical_method("hantush jacob 1955") == HANTUSH_JACOB
        assert canonical_method("Cooper-Jacob") == COOPER_JACOB

    def test_canonical_labels_match_the_dropdown(self):
        from src.agents.method_selection import HANTUSH_JACOB

        assert SUPPORTED_METHODS == [THEIS, COOPER_JACOB, PAPADOPULOS_COOPER,
                                     HANTUSH_JACOB]


# ── Well geometry parsing ─────────────────────────────────────────────────────

class TestExtractWellRadius:

    @pytest.mark.parametrize("text,expected", [
        ("the well has a radius of 0.5 m", 0.5),
        ("radius 0.25 metres", 0.25),
        ("a 1.2 m diameter well", 0.6),
        ("diameter of 2 m", 1.0),
        ("well diameter 60 cm", 0.30),
        ("radius = 150 mm", 0.150),
    ])
    def test_parses_stated_dimensions(self, text, expected):
        assert extract_well_radius_m(text) == pytest.approx(expected)

    def test_diameter_is_halved(self):
        assert extract_well_radius_m("1.0 m diameter") == pytest.approx(0.5)
        assert extract_well_radius_m("1.0 m radius") == pytest.approx(1.0)

    def test_largest_dimension_wins(self):
        text = "a 2 m diameter caisson with a 0.1 m radius screen"
        assert extract_well_radius_m(text) == pytest.approx(1.0)

    @pytest.mark.parametrize("text", ["", "no numbers here", "pumped at 500 m3/day"])
    def test_returns_none_when_nothing_stated(self, text):
        assert extract_well_radius_m(text) is None

    def test_distance_to_observation_well_is_not_a_radius(self):
        """'50 m away' is a distance, not a well radius."""
        assert extract_well_radius_m("observation well 50 m away") is None


# ── Rule 1: explicit method wins ──────────────────────────────────────────────

class TestExplicitMethod:

    def test_named_method_is_honoured(self):
        s = suggest_method(aquifer_context="confined", selected_method="Theis")

        assert s["method"] == THEIS
        assert s["confidence"] >= 0.9
        assert s["supported"] is True

    def test_named_method_in_free_text_is_honoured(self):
        s = suggest_method(user_message="can you run Cooper-Jacob on my data?")
        assert s["method"] == COOPER_JACOB

    def test_explicit_method_overrides_geometry_cues(self):
        """The user knows their own setup better than a keyword heuristic."""
        s = suggest_method(
            aquifer_context="confined",
            selected_method="Theis",
            user_message="my large-diameter dug well, 1 m radius",
        )
        assert s["method"] == THEIS

    def test_reason_mentions_the_method(self):
        s = suggest_method(selected_method="Papadopulos-Cooper")
        assert PAPADOPULOS_COOPER in s["reason"]


# ── Rule 2: unsupported aquifers ──────────────────────────────────────────────

class TestUnsupportedAquifers:

    @pytest.mark.parametrize("aquifer", ["unconfined", "fractured"])
    def test_no_method_offered(self, aquifer):
        s = suggest_method(aquifer_context=aquifer)

        assert s["method"] == ""
        assert s["supported"] is False
        assert aquifer in s["reason"]

    def test_reason_says_what_is_available(self):
        s = suggest_method(aquifer_context="unconfined")
        assert "confined" in s["reason"]

    def test_explicit_method_still_wins_over_aquifer_gap(self):
        """If they explicitly ask for Theis on an unconfined test, let them."""
        s = suggest_method(aquifer_context="unconfined", selected_method="Theis")
        assert s["method"] == THEIS


# ── Rule 3: large-diameter wells ──────────────────────────────────────────────

class TestLargeDiameterWell:

    @pytest.mark.parametrize("text", [
        "I pumped a large-diameter dug well",
        "this is an open well in the village",
        "wellbore storage dominates the early data",
        "drawdown was measured in the pumped well itself",
        "a caisson well",
    ])
    def test_phrase_cues_select_papadopulos_cooper(self, text):
        s = suggest_method(aquifer_context="confined", user_message=text)
        assert s["method"] == PAPADOPULOS_COOPER

    def test_stated_radius_above_threshold_selects_papadopulos_cooper(self):
        s = suggest_method(aquifer_context="confined",
                           user_message="the well has a radius of 0.5 m")

        assert s["method"] == PAPADOPULOS_COOPER
        assert "0.50 m" in s["reason"]

    def test_small_radius_does_not_trigger_it(self):
        s = suggest_method(aquifer_context="confined",
                           user_message="the well has a radius of 0.05 m")
        assert s["method"] == THEIS

    def test_threshold_boundary_is_exclusive(self):
        """At exactly the threshold, wellbore storage is still treated as minor."""
        at = suggest_method(
            aquifer_context="confined",
            user_message=f"radius of {LARGE_WELL_RADIUS_M} m")
        assert at["method"] == THEIS

    def test_reason_cites_the_theis_assumption(self):
        s = suggest_method(aquifer_context="confined",
                           user_message="large-diameter well")
        assert "infinitesimal" in s["reason"]

    def test_reason_carries_the_storativity_caveat(self):
        """Paper p. 244 — S is only order-of-magnitude reliable."""
        s = suggest_method(aquifer_context="confined",
                           user_message="large-diameter well")
        assert "order of magnitude" in s["reason"]


# ── Rule 4: late-time framing ─────────────────────────────────────────────────

class TestLateTimeFraming:

    @pytest.mark.parametrize("text", [
        "I want the late-time straight line",
        "can you do a semi-log analysis",
        "fit the straight-line portion",
        "how many log cycles do I need",
    ])
    def test_selects_cooper_jacob(self, text):
        s = suggest_method(aquifer_context="confined", user_message=text)
        assert s["method"] == COOPER_JACOB

    def test_reason_cites_the_u_threshold(self):
        s = suggest_method(aquifer_context="confined",
                           user_message="late-time straight line")
        assert "0.02" in s["reason"]

    def test_large_well_takes_precedence_over_late_time(self):
        """Wellbore storage invalidates the line-source assumption entirely."""
        s = suggest_method(
            aquifer_context="confined",
            user_message="late-time straight line for my large-diameter dug well")
        assert s["method"] == PAPADOPULOS_COOPER


# ── Rule 5: default ───────────────────────────────────────────────────────────

class TestDefault:

    def test_confined_defaults_to_theis(self):
        s = suggest_method(aquifer_context="confined",
                           user_message="analyse my pumping test")

        assert s["method"] == THEIS
        assert s["supported"] is True

    def test_unknown_aquifer_lowers_confidence(self):
        confined = suggest_method(aquifer_context="confined")
        unknown = suggest_method(aquifer_context="unknown")

        assert unknown["confidence"] < confined["confidence"]

    def test_unknown_aquifer_asks_for_confirmation(self):
        s = suggest_method(aquifer_context="unknown")
        assert "confirm" in s["reason"].lower()

    def test_every_suggestion_has_the_full_card_shape(self):
        for kwargs in [{"aquifer_context": "confined"},
                       {"aquifer_context": "unconfined"},
                       {"selected_method": "Theis"},
                       {"user_message": "large-diameter well"}]:
            s = suggest_method(**kwargs)
            assert set(s) == {"method", "reason", "confidence", "supported"}
            assert 0.0 <= s["confidence"] <= 1.0
            assert s["reason"]

    def test_suggested_method_is_always_implemented(self):
        for kwargs in [{"aquifer_context": "confined"},
                       {"user_message": "semi-log"},
                       {"user_message": "dug well"}]:
            s = suggest_method(**kwargs)
            assert s["method"] in SUPPORTED_METHODS


# ── Rendering ─────────────────────────────────────────────────────────────────

class TestFormatSuggestion:

    def test_supported_suggestion_renders_method_and_reason(self):
        out = format_suggestion(suggest_method(aquifer_context="confined"))

        assert THEIS in out
        assert "Confidence" in out

    def test_unsupported_suggestion_renders_the_explanation_only(self):
        out = format_suggestion(suggest_method(aquifer_context="unconfined"))

        assert "unconfined" in out
        assert "Confidence" not in out

    def test_empty_suggestion_renders_empty(self):
        assert format_suggestion({}) == ""


class TestSummariseResult:

    def _result(self, **over):
        base = {
            "method": "Theis (1935)", "T": 2.49e-3, "S": 5.05e-5,
            "T_day": 215.29, "rmse": 0.0042, "r_squared": 0.9999,
            "validity_notes": ["note one", "note two"], "success": True,
        }
        base.update(over)
        return base

    def test_reports_the_key_parameters(self):
        out = summarise_result(self._result())

        assert "215.29" in out       # T in m²/day
        assert "5.05e-05" in out     # S
        assert "0.9999" in out       # R²

    def test_includes_test_setup_when_given(self):
        out = summarise_result(self._result(),
                               {"Q": 500 / 86400, "r": 50.0, "n_points": 60})

        assert "500.0 m³/day" in out
        assert "50.0 m" in out
        assert "60 points" in out

    def test_lists_validity_notes(self):
        out = summarise_result(self._result())
        assert "note one" in out

    def test_truncates_long_note_lists(self):
        out = summarise_result(self._result(
            validity_notes=[f"n{i}" for i in range(9)]))

        assert "5 more" in out

    def test_failed_result_reports_the_error(self):
        out = summarise_result(self._result(success=False,
                                            error_message="did not converge"))

        assert "did not converge" in out

    def test_empty_result_renders_empty(self):
        assert summarise_result({}) == ""


# ── Rule 2: the data outranks the prose ───────────────────────────────────────

class TestDataDrivenSelection:
    """The Bourdet derivative reading the flow regime off the actual series.

    This is what separates a recommender that pattern-matches the user's wording
    from one that reads their data. The decisive case is the disagreement test:
    a user describing a "confined aquifer test" whose curve plainly shows
    leakage should be steered to Hantush-Jacob, not Theis.
    """

    Q, R, T, S = 500 / 86400, 50.0, 2.5e-3, 5e-5

    def _theis(self, t):
        import numpy as np
        from scipy.special import expn
        u = (self.R ** 2 * self.S) / (4.0 * self.T * t)
        return (self.Q / (4.0 * np.pi * self.T)) * expn(1, np.maximum(u, 1e-12))

    def _leaky(self, t, B=300.0):
        from src.calculator.methods.hantush_jacob import hantush_jacob_drawdown
        return hantush_jacob_drawdown(t, self.T, self.S, B, self.Q, self.R)

    def test_radial_data_confirms_theis(self):
        import numpy as np
        t = np.logspace(2, 6, 60)
        s = suggest_method(aquifer_context="confined", time_s=t,
                           drawdown_m=self._theis(t), Q=self.Q)

        assert s["method"] == THEIS
        assert "radial flow" in s["reason"]

    def test_the_reason_quotes_the_transmissivity_read_off_the_plateau(self):
        """An independent T that owes nothing to curve fitting."""
        import numpy as np
        t = np.logspace(3, 7, 60)
        s = suggest_method(aquifer_context="confined", time_s=t,
                           drawdown_m=self._theis(t), Q=self.Q)

        assert "216.0 m²/day" in s["reason"]

    def test_leaky_data_overrides_a_confined_description(self):
        """The case the whole feature exists for."""
        import numpy as np
        from src.agents.method_selection import HANTUSH_JACOB

        t = np.logspace(2, 7, 60)
        prose_only = suggest_method(
            aquifer_context="confined",
            user_message="here is my confined aquifer pumping test")
        with_data = suggest_method(
            aquifer_context="confined",
            user_message="here is my confined aquifer pumping test",
            time_s=t, drawdown_m=self._leaky(t), Q=self.Q)

        assert prose_only["method"] == THEIS
        assert with_data["method"] == HANTUSH_JACOB

    def test_wellbore_storage_data_selects_papadopulos_cooper(self):
        import numpy as np
        from src.calculator.papadopulos_cooper import pc_drawdown

        t = np.logspace(0, 5, 60)
        s = pc_drawdown(t, 1e-3, 1e-4, 300 / 86400, 0.30, 0.20)
        card = suggest_method(aquifer_context="confined", time_s=t,
                              drawdown_m=s, Q=300 / 86400)

        assert card["method"] == PAPADOPULOS_COOPER

    def test_an_explicitly_named_method_still_wins(self):
        """The user knows their own test setup; data evidence does not override
        a direct instruction."""
        import numpy as np
        t = np.logspace(2, 7, 60)
        card = suggest_method(aquifer_context="confined", selected_method="Theis",
                              time_s=t, drawdown_m=self._leaky(t), Q=self.Q)

        assert card["method"] == THEIS

    def test_data_beats_a_misleading_keyword_cue(self):
        """"dug well" would normally select Papadopulos-Cooper, but a curve with
        no wellbore-storage signature says otherwise."""
        import numpy as np
        t = np.logspace(2, 6, 60)
        card = suggest_method(aquifer_context="confined",
                              user_message="my dug well test",
                              time_s=t, drawdown_m=self._theis(t), Q=self.Q)

        assert card["method"] == THEIS

    def test_evidence_is_attached_when_the_data_was_consulted(self):
        import numpy as np
        t = np.logspace(2, 6, 60)
        card = suggest_method(aquifer_context="confined", time_s=t,
                              drawdown_m=self._theis(t), Q=self.Q)

        assert "evidence" in card and card["evidence"]

    def test_no_evidence_key_when_only_prose_was_used(self):
        card = suggest_method(aquifer_context="confined",
                              user_message="analyse my test")
        assert "evidence" not in card

    def test_unrecognisable_regime_warns_rather_than_recommending(self):
        """A barrier has no implemented method. Falling through to Theis would
        fit a model the data has just contradicted."""
        import numpy as np
        t = np.logspace(1, 5, 50)
        card = suggest_method(aquifer_context="confined", time_s=t,
                              drawdown_m=0.2 * np.log(t) + 3e-5 * t, Q=self.Q)

        assert card["method"] == ""
        assert card["supported"] is False
        assert "boundary" in card["reason"]

    def test_too_few_points_falls_back_to_the_keyword_rules(self):
        import numpy as np
        t = np.array([10.0, 20.0, 30.0])
        card = suggest_method(aquifer_context="confined", time_s=t,
                              drawdown_m=np.array([0.1, 0.2, 0.3]), Q=self.Q)

        assert card["method"] == THEIS
        assert "evidence" not in card

    def test_malformed_series_never_breaks_selection(self):
        """A bad upload must not take down method selection."""
        import numpy as np
        card = suggest_method(aquifer_context="confined",
                              time_s=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                              drawdown_m=np.array([1.0, 2.0]), Q=self.Q)

        assert card["method"] == THEIS

    def test_missing_pumping_rate_still_diagnoses(self):
        import numpy as np
        t = np.logspace(2, 6, 60)
        card = suggest_method(aquifer_context="confined", time_s=t,
                              drawdown_m=self._theis(t))

        assert card["method"] == THEIS
