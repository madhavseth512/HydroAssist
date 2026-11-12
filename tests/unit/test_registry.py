"""Integrity tests for the method registry.

The registry is about to become the single source of truth that the UI, the chat
method-selection layer and the data validator all read from. That makes a handful of
its properties load-bearing in ways that are easy to break silently later:

  * display order (an existing test asserts an exact three-element list)
  * alias resolution, where "Papadopulos-Cooper" and "Cooper-Jacob" share the word
    "cooper" and the longer alias must win
  * stdlib-only imports, so non-numeric modules can import it cheaply
  * every declared factory actually resolving to a working calculator
"""

import pytest

from src.calculator import registry as R


# ── Table integrity ───────────────────────────────────────────────────────────

class TestTableIntegrity:

    def test_ids_are_unique(self):
        assert len(R.ids(implemented_only=False)) == len(set(R.ids(implemented_only=False)))

    def test_labels_are_unique(self):
        labels = R.labels(implemented_only=False)
        assert len(labels) == len(set(labels))

    def test_every_spec_key_matches_its_id(self):
        for key, spec in R.REGISTRY.items():
            assert key == spec.id

    def test_display_order_groups_by_aquifer_then_year(self):
        """Dropdown order: all the confined methods, then leaky, and so on, each
        group chronological. Grouping beats a global date sort because a user
        picks an aquifer type first."""
        seen_groups, years_in_group, previous = [], [], None

        for spec in R.all_specs(implemented_only=False):
            group = sorted(spec.aquifer_types)[0]
            if group != previous:
                assert group not in seen_groups, (
                    f"{group} methods are split across the table")
                seen_groups.append(group)
                years_in_group = []
                previous = group
            years_in_group.append(spec.year)
            assert years_in_group == sorted(years_in_group), (
                f"{group} methods are not in chronological order")

    def test_current_methods_and_order(self):
        """Load-bearing: the dropdown and several tests depend on this order."""
        assert R.labels() == [
            "Theis (1935)",
            "Theis Recovery (1935)",
            "Cooper-Jacob (1946)",
            "Papadopulos-Cooper (1967)",
            "Hantush-Jacob (1955)",
            "Boulton (1963)",
            "Warren-Root (1963)",
            "Ogata-Banks (1961)",
            "Thiéry (2022)",
        ]

    @pytest.mark.parametrize("spec", R.all_specs(implemented_only=False),
                             ids=lambda s: s.id)
    def test_spec_is_fully_populated(self, spec):
        assert spec.label and spec.citation and spec.ui_help
        assert spec.authors and spec.parameters
        assert spec.aquifer_types <= set(R.AQUIFER_TYPES)
        assert spec.test_types <= set(R.TEST_TYPES)
        assert spec.data_source in (R.OBSERVATION_WELL, R.PUMPED_WELL, R.EITHER)

    @pytest.mark.parametrize("spec", R.all_specs(implemented_only=False),
                             ids=lambda s: s.id)
    def test_citation_names_the_year(self, spec):
        assert str(spec.year) in spec.citation

    def test_spec_is_immutable(self):
        """Specs are shared global state; accidental mutation would be invisible."""
        with pytest.raises(Exception):
            R.REGISTRY["theis_1935"].label = "something else"


# ── Factories ─────────────────────────────────────────────────────────────────

class TestFactories:

    @pytest.mark.parametrize("spec", R.all_specs(), ids=lambda s: s.id)
    def test_implemented_method_builds_a_calculator(self, spec):
        from src.calculator.base import BaseCalculator
        calc = spec.build()
        assert isinstance(calc, BaseCalculator)

    @pytest.mark.parametrize("spec", R.all_specs(), ids=lambda s: s.id)
    def test_built_calculator_reports_the_registered_label(self, spec):
        """The calculator's own `method` string must match the registry label,
        or results will not round-trip through AgentState."""
        import pandas as pd
        from src.calculator.base import CalculationInput

        df = pd.read_csv("tests/theis_cooper_jacob_sample.csv")
        kwargs = dict(df=df, Q=500 / 86400, r=50.0)
        if spec.data_source == R.PUMPED_WELL:
            kwargs.update(r=0.30, r_w=0.30, r_c=0.20)

        result = spec.build().calculate(CalculationInput(**kwargs))
        assert result.method == spec.label

    def test_build_helper_accepts_id_or_label(self):
        from src.calculator.theis import TheisCalculator
        assert isinstance(R.build("theis_1935"), TheisCalculator)
        assert isinstance(R.build("Theis (1935)"), TheisCalculator)

    def test_unimplemented_method_raises_not_implemented(self):
        spec = R.MethodSpec(
            id="x", label="X (1999)", year=1999, authors=("X",), citation="X 1999",
            aquifer_types=frozenset({R.CONFINED}),
            test_types=frozenset({R.CONSTANT_RATE}),
            parameters=("T",), data_source=R.EITHER, implemented=False,
        )
        with pytest.raises(NotImplementedError, match="no implementation"):
            spec.build()


# ── Lookup and free-text resolution ───────────────────────────────────────────

class TestLookup:

    def test_get_by_id_and_by_label(self):
        assert R.get("theis_1935").id == "theis_1935"
        assert R.get("Theis (1935)").id == "theis_1935"

    @pytest.mark.parametrize("bad", ["", None, "Neuman (1974)", "nonsense"])
    def test_get_returns_none_for_unknown(self, bad):
        assert R.get(bad) is None

    def test_require_raises_with_a_useful_message(self):
        with pytest.raises(KeyError) as exc:
            R.require("Neuman (1974)")
        assert "Theis (1935)" in str(exc.value), "the error should list what IS known"

    @pytest.mark.parametrize("text,expected", [
        ("Theis", "theis_1935"),
        ("theis", "theis_1935"),
        ("the theis method", "theis_1935"),
        ("Theis (1935)", "theis_1935"),
        ("Cooper-Jacob", "cooper_jacob_1946"),
        ("cooper jacob", "cooper_jacob_1946"),
        ("Cooper_Jacob (1946)", "cooper_jacob_1946"),
        ("use the jacob straight line", "cooper_jacob_1946"),
        ("Papadopulos-Cooper", "papadopulos_cooper_1967"),
        ("papadopulos cooper 1967", "papadopulos_cooper_1967"),
        ("papadopulos", "papadopulos_cooper_1967"),
    ])
    def test_canonical_id_resolves_free_text(self, text, expected):
        assert R.canonical_id(text) == expected

    def test_longer_alias_wins_on_shared_surname(self):
        """Both methods contain 'Cooper'; the more specific name must win.
        This is the single most breakable rule in the module."""
        assert R.canonical_id("Papadopulos-Cooper (1967)") == "papadopulos_cooper_1967"
        assert R.canonical_id("run papadopulos cooper") == "papadopulos_cooper_1967"

    @pytest.mark.parametrize("text", ["", None, "Neuman", "Moench",
                                      "Streltsova", "xyz"])
    def test_canonical_id_returns_none_for_unimplemented_methods(self, text):
        """Real-but-absent methods must not be silently coerced onto a present one."""
        assert R.canonical_id(text) is None

    def test_canonical_label_round_trips(self):
        for label in R.labels():
            assert R.canonical_label(label) == label


# ── Filtering ─────────────────────────────────────────────────────────────────

class TestFiltering:

    def test_confined_methods(self):
        assert len(R.for_aquifer(R.CONFINED)) == 4

    @pytest.mark.parametrize("aquifer,expected", [
        (R.LEAKY, "hantush_jacob_1955"),
        (R.UNCONFINED, "boulton_1963"),
        (R.FRACTURED, "warren_root_1963"),
    ])
    def test_each_non_confined_type_has_its_method(self, aquifer, expected):
        assert [s.id for s in R.for_aquifer(aquifer)] == [expected]

    def test_every_aquifer_type_is_now_covered(self):
        """The coverage half of the plan's minimum-credible-alternative bar:
        one canonical method per aquifer type."""
        assert R.supported_aquifers() == set(R.AQUIFER_TYPES)

    def test_every_method_but_recovery_and_tracer_is_constant_rate(self):
        assert len(R.for_test_type(R.CONSTANT_RATE)) == len(R.all_specs()) - 3

    def test_recovery_has_its_own_method(self):
        assert [s.id for s in R.for_test_type(R.RECOVERY)] == ["theis_recovery_1935"]

    @pytest.mark.parametrize("test_type", [R.STEP_DRAWDOWN, R.SLUG])
    def test_no_methods_yet_for_other_test_types(self, test_type):
        assert R.for_test_type(test_type) == []

    def test_papadopulos_cooper_is_the_only_pumped_well_method(self):
        pumped = [s.id for s in R.all_specs() if s.data_source == R.PUMPED_WELL]
        assert pumped == ["papadopulos_cooper_1967"]

    def test_papadopulos_cooper_declares_its_geometry_requirement(self):
        spec = R.get("papadopulos_cooper_1967")
        assert "well.r_w" in spec.requires and "well.r_c" in spec.requires

    def test_methods_needing_no_extra_geometry_declare_none(self):
        for method_id in ("theis_1935", "cooper_jacob_1946"):
            assert R.get(method_id).requires == ()


# ── Import weight ─────────────────────────────────────────────────────────────

class TestImportsAreCheap:

    def test_registry_pulls_in_no_numeric_stack(self):
        """validator.py and method_selection.py import this and do no numerics.
        Keeping it stdlib-only also avoids a circular import with plots.py."""
        import subprocess
        import sys

        code = (
            "import sys; import src.calculator.registry; "
            "heavy = {'scipy','numpy','pandas','matplotlib','plotly','streamlit'} "
            "& {m.split('.')[0] for m in sys.modules}; "
            "print(sorted(heavy))"
        )
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "[]", f"registry pulled in {out.stdout.strip()}"


class TestAMisspeltMethodName:
    """A method is named after the people who published it, and those names are
    hard to spell.

    Reported from the deployed site: a question about Papadopulos-Cooper came
    back as Cooper-Jacob. The exact-alias reading returned None for the spelling
    used, so the caller fell back to the language model's extraction — and that
    was wrong. Reading a near-miss here removes the fallback.
    """

    @pytest.mark.parametrize("text,expected", [
        ("papadulous-cooper", "papadopulos_cooper_1967"),
        ("papadopolous cooper", "papadopulos_cooper_1967"),
        ("tell me about papadulous cooper for a large well",
         "papadopulos_cooper_1967"),
        ("coper-jacob", "cooper_jacob_1946"),
        ("theiss method", "theis_1935"),
        ("hantush jakob", "hantush_jacob_1955"),
        ("boluton", "boulton_1963"),
    ])
    def test_a_near_miss_still_resolves(self, text, expected):
        assert R.canonical_id(text) == expected

    def test_an_exact_name_is_unaffected(self):
        assert R.canonical_id("papadopulos-cooper") == \
            "papadopulos_cooper_1967"
        assert R.canonical_id("cooper-jacob") == "cooper_jacob_1946"

    @pytest.mark.parametrize("text", [
        "cooper",                      # shared by two methods, so still ambiguous
        "what is transmissivity",
        "the aquifer is confined",
        "my drawdown readings look odd",
        "",
    ])
    def test_it_does_not_invent_a_method(self, text):
        """A wrong method is worse than no method: it produces a plausible fit
        and a transmissivity nobody asked for."""
        assert R.canonical_id(text) is None

    def test_two_names_spelled_about_as_nearly_resolve_to_neither(self):
        """Where the text is as close to one method as to another, guessing
        between them would be a coin toss with a number on the end of it."""
        # "cooper jacob" against "papadopulos cooper" and "cooper-jacob": the
        # exact alias wins here, so the guard is tested on its own terms.
        from src.calculator.registry import _nearly

        assert _nearly("cooper") is None

    def test_a_short_word_is_not_treated_as_a_name(self):
        from src.calculator.registry import _nearly

        assert _nearly("the") is None
        assert _nearly("a b c") is None
