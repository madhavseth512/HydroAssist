"""The single source of truth for which analysis methods exist.

Before this module the literal string "Theis (1935)" appeared in ten files — app.py,
both agents, the data validator, and every calculator, plot and report module — so
adding a method meant finding all ten. Everything that needs to enumerate, resolve or
dispatch a method now reads it from here instead.

Two deliberate design choices:

* **One declarative table, not decorator self-registration.** You can read this file
  top to bottom and see every method the product supports. Decorators would make the
  contents depend on import order, which is exactly the kind of thing that produces
  "why isn't my method in the dropdown".

* **Lazy factories, stdlib-only imports.** Nothing here imports scipy, numpy or the
  calculator modules at module scope. That lets `src/data/validator.py` and
  `src/agents/method_selection.py` — neither of which does any numerics — import the
  registry cheaply, and it removes any circular-import risk with the plot modules
  that already import from the calculators.

Adding a method is one `MethodSpec` entry plus its implementation module.
"""

from dataclasses import dataclass
from typing import Callable, Optional


# ── Vocabularies ──────────────────────────────────────────────────────────────
# Plain string constants rather than enums, because these values are already
# persisted in AgentState and compared against LLM output. Enums would force a
# serialisation layer for no benefit at this size.

CONFINED = "confined"
LEAKY = "leaky"
UNCONFINED = "unconfined"
FRACTURED = "fractured"

AQUIFER_TYPES = (CONFINED, LEAKY, UNCONFINED, FRACTURED)

CONSTANT_RATE = "constant_rate"
RECOVERY = "recovery"
STEP_DRAWDOWN = "step_drawdown"
SLUG = "slug"
CONSTANT_HEAD = "constant_head"

TEST_TYPES = (CONSTANT_RATE, RECOVERY, STEP_DRAWDOWN, SLUG, CONSTANT_HEAD)

# Where the drawdown series was measured. Papadopulos-Cooper is fitted at the
# pumped well face; Theis and Cooper-Jacob assume an observation well at distance r.
OBSERVATION_WELL = "observation_well"
PUMPED_WELL = "pumped_well"
EITHER = "either"


@dataclass(frozen=True)
class PlotContext:
    """The test parameters a plot needs, in one shape for every method.

    The underlying plot functions take different arguments — Theis needs Q and r,
    Cooper-Jacob also needs the excluded-point count, Papadopulos-Cooper needs the
    two well radii instead of a distance. Passing this single object lets the
    registry dispatch them uniformly; each adapter picks out what its method uses.
    """

    Q: float                    # pumping rate [m^3/s]
    r: float = 0.0              # observation distance [m]
    r_w: float = 0.10           # well screen radius [m]
    r_c: float = 0.10           # casing radius [m]
    n_excluded: int = 0         # early-time points dropped by the u-filter (CJ)


@dataclass(frozen=True)
class MethodSpec:
    """Everything the rest of the app needs to know about one analysis method.

    `label` is the canonical display string and doubles as the persisted key in
    AgentState and the Streamlit dropdown, so it must stay stable once shipped.
    `id` is the stable machine key — prefer it for new code, filenames and URLs.
    """

    id: str
    label: str
    year: int
    authors: tuple
    citation: str

    aquifer_types: frozenset
    test_types: frozenset
    parameters: tuple           # canonical names of what it estimates: ("T", "S")
    data_source: str            # OBSERVATION_WELL | PUMPED_WELL | EITHER

    # Free-text fragments the chat layer matches against, lowercase. Order does not
    # matter here — canonical_id() sorts by descending length so that the more
    # specific "papadopulos" always beats the shared "cooper".
    aliases: tuple = ()

    # Dotted geometry paths this method cannot run without, e.g. ("well.r_w",).
    # Currently informational; the base calculator will enforce it once the
    # geometry dataclasses land.
    requires: tuple = ()

    implemented: bool = True

    # Lazy — each does a local import so this module stays stdlib-only.
    factory: Optional[Callable] = None

    # Plot adapters, all taking (result, PlotContext). Lazy for the same reason.
    diagnostic_plot: Optional[Callable] = None   # -> plotly Figure
    report_plot: Optional[Callable] = None       # -> matplotlib Figure, for the PDF
    # Manual-matching plot: (result, ctx, T_manual, S_manual) -> (Figure, rmse_mm)
    interactive_plot: Optional[Callable] = None

    ui_help: str = ""
    selection_rationale: str = ""

    def supports_aquifer(self, aquifer: str) -> bool:
        return aquifer in self.aquifer_types

    def supports_test_type(self, test_type: str) -> bool:
        return test_type in self.test_types

    def build(self):
        """Instantiate the calculator. Raises if the method is not implemented."""
        if not self.implemented or self.factory is None:
            raise NotImplementedError(
                f"{self.label} is listed in the registry but has no implementation yet."
            )
        return self.factory()


# ── Lazy factories ────────────────────────────────────────────────────────────

def _theis():
    from src.calculator.theis import TheisCalculator
    return TheisCalculator()


def _cooper_jacob():
    from src.calculator.cooper_jacob import CooperJacobCalculator
    return CooperJacobCalculator()


def _papadopulos_cooper():
    from src.calculator.papadopulos_cooper import PapadopulosCooperCalculator
    return PapadopulosCooperCalculator()


def _hantush_jacob():
    from src.calculator.methods.hantush_jacob import HantushJacobCalculator
    return HantushJacobCalculator()


# ── Lazy plot adapters ────────────────────────────────────────────────────────
# Each normalises one method's plot signature onto (result, PlotContext), so the
# UI and the PDF builder can dispatch without knowing which method they hold.

def _theis_diagnostic(result, ctx):
    from src.calculator.plots import theis_diagnostic
    return theis_diagnostic(result, ctx.Q, ctx.r)


def _cj_diagnostic(result, ctx):
    from src.calculator.plots import cj_diagnostic
    return cj_diagnostic(result, ctx.Q, ctx.r, ctx.n_excluded)


def _pc_diagnostic(result, ctx):
    from src.calculator.plots import pc_diagnostic
    return pc_diagnostic(result, ctx.Q, ctx.r_w, ctx.r_c)


def _theis_interactive(result, ctx, T_manual, S_manual):
    from src.calculator.plots import theis_interactive
    return theis_interactive(result.time_s, result.drawdown_obs, ctx.Q, ctx.r,
                             result.T, result.S, T_manual, S_manual)


def _cj_interactive(result, ctx, T_manual, S_manual):
    from src.calculator.plots import cj_interactive
    return cj_interactive(result.time_s, result.drawdown_obs, ctx.Q, ctx.r,
                          result.T, result.S, T_manual, S_manual)


def _pc_interactive(result, ctx, T_manual, S_manual):
    from src.calculator.plots import pc_interactive
    return pc_interactive(result.time_s, result.drawdown_obs,
                          ctx.Q, ctx.r_w, ctx.r_c,
                          result.T, result.S, T_manual, S_manual)


def _theis_report_plot(result, ctx):
    from src.calculator.report import _theis_mpl
    return _theis_mpl(result, ctx.Q, ctx.r)


def _cj_report_plot(result, ctx):
    from src.calculator.report import _cj_mpl
    return _cj_mpl(result, ctx.n_excluded)


def _pc_report_plot(result, ctx):
    from src.calculator.report import _pc_mpl
    return _pc_mpl(result, ctx.Q, ctx.r_w, ctx.r_c)


# ── The table ─────────────────────────────────────────────────────────────────
# Insertion order is the display order, and it is load-bearing: an existing test
# asserts the supported-method list equals [Theis, Cooper-Jacob, Papadopulos-Cooper]
# in exactly this order.

_SPECS = (
    MethodSpec(
        id="theis_1935",
        label="Theis (1935)",
        year=1935,
        authors=("Theis, C.V.",),
        citation=(
            "Theis, C.V. (1935). The relation between the lowering of the "
            "piezometric surface and the rate and duration of discharge of a well "
            "using ground-water storage. Transactions AGU, 16, 519-524."
        ),
        aquifer_types=frozenset({CONFINED}),
        test_types=frozenset({CONSTANT_RATE}),
        parameters=("T", "S"),
        data_source=OBSERVATION_WELL,
        aliases=("theis",),
        factory=_theis,
        diagnostic_plot=_theis_diagnostic,
        interactive_plot=_theis_interactive,
        report_plot=_theis_report_plot,
        ui_help="Full-curve type-curve fit. Valid at all times; the general default.",
        selection_rationale=(
            "Theis (1935) is the general full-curve method for a confined aquifer "
            "and uses every data point, so it is the safe default."
        ),
    ),
    MethodSpec(
        id="cooper_jacob_1946",
        label="Cooper-Jacob (1946)",
        year=1946,
        authors=("Cooper, H.H.", "Jacob, C.E."),
        citation=(
            "Cooper, H.H. & Jacob, C.E. (1946). A generalized graphical method for "
            "evaluating formation constants and summarizing well-field history. "
            "Transactions AGU, 27(4), 526-534."
        ),
        aquifer_types=frozenset({CONFINED}),
        test_types=frozenset({CONSTANT_RATE}),
        parameters=("T", "S"),
        data_source=OBSERVATION_WELL,
        aliases=("cooper-jacob", "cooper jacob", "cooper_jacob", "jacob"),
        factory=_cooper_jacob,
        diagnostic_plot=_cj_diagnostic,
        interactive_plot=_cj_interactive,
        report_plot=_cj_report_plot,
        ui_help="Late-time semi-log straight line, valid where u < 0.02.",
        selection_rationale=(
            "Cooper-Jacob (1946) fits a line to s vs log t where u < 0.02 "
            "(paper p. 3); early-time points are excluded automatically."
        ),
    ),
    MethodSpec(
        id="papadopulos_cooper_1967",
        label="Papadopulos-Cooper (1967)",
        year=1967,
        authors=("Papadopulos, I.S.", "Cooper, H.H."),
        citation=(
            "Papadopulos, I.S. & Cooper, H.H. (1967). Drawdown in a well of large "
            "diameter. Water Resources Research, 3(1), 241-244."
        ),
        aquifer_types=frozenset({CONFINED}),
        test_types=frozenset({CONSTANT_RATE}),
        parameters=("T", "S"),
        data_source=PUMPED_WELL,
        aliases=("papadopulos-cooper", "papadopulos cooper", "papadopulos"),
        factory=_papadopulos_cooper,
        diagnostic_plot=_pc_diagnostic,
        interactive_plot=_pc_interactive,
        report_plot=_pc_report_plot,
        requires=("well.r_w", "well.r_c"),
        ui_help="Large-diameter well with wellbore storage; data from the pumped well.",
        selection_rationale=(
            "Papadopulos-Cooper (1967) accounts for wellbore storage in a "
            "large-diameter well, which Theis assumption (5) — a well of "
            "infinitesimal diameter — does not."
        ),
    ),
    MethodSpec(
        id="hantush_jacob_1955",
        label="Hantush-Jacob (1955)",
        year=1955,
        authors=("Hantush, M.S.", "Jacob, C.E."),
        citation=(
            "Hantush, M.S. & Jacob, C.E. (1955). Non-steady radial flow in an "
            "infinite leaky aquifer. Transactions AGU, 36(1), 95-100."
        ),
        aquifer_types=frozenset({LEAKY}),
        test_types=frozenset({CONSTANT_RATE}),
        parameters=("T", "S", "B"),
        data_source=OBSERVATION_WELL,
        aliases=("hantush-jacob", "hantush jacob", "hantush"),
        factory=_hantush_jacob,
        ui_help=(
            "Leaky confined aquifer: drawdown stabilises as water leaks through "
            "the aquitard. Estimates T, S and the leakage factor B."
        ),
        selection_rationale=(
            "Hantush-Jacob (1955) accounts for vertical leakage through a "
            "confining layer, which makes drawdown level off at late time "
            "instead of growing indefinitely as Theis predicts."
        ),
    ),
)

REGISTRY = {spec.id: spec for spec in _SPECS}


# ── Lookup ────────────────────────────────────────────────────────────────────

def all_specs(implemented_only: bool = True) -> list:
    """Every method, in display order."""
    return [s for s in REGISTRY.values() if s.implemented or not implemented_only]


def labels(implemented_only: bool = True) -> list:
    """Canonical display labels, in display order — the dropdown contents."""
    return [s.label for s in all_specs(implemented_only)]


def ids(implemented_only: bool = True) -> list:
    return [s.id for s in all_specs(implemented_only)]


def get(id_or_label: str) -> Optional[MethodSpec]:
    """Resolve an exact id or an exact label. Use canonical_id() for free text."""
    if not id_or_label:
        return None
    if id_or_label in REGISTRY:
        return REGISTRY[id_or_label]
    for spec in REGISTRY.values():
        if spec.label == id_or_label:
            return spec
    return None


def require(id_or_label: str) -> MethodSpec:
    """get(), but raises with a useful message instead of returning None."""
    spec = get(id_or_label)
    if spec is None:
        raise KeyError(
            f"Unknown analysis method {id_or_label!r}. "
            f"Known methods: {', '.join(labels(implemented_only=False))}"
        )
    return spec


def canonical_id(text: str) -> Optional[str]:
    """Resolve free text ("the theis method", "Cooper_Jacob") to a method id.

    Aliases are tried longest-first so a more specific name wins: both
    "Papadopulos-Cooper" and "Cooper-Jacob" contain "cooper", and the longer
    alias must match first. Exact ids and labels are checked before aliases.
    """
    if not text:
        return None

    spec = get(text)
    if spec is not None:
        return spec.id

    lowered = text.lower()
    candidates = []
    for spec in REGISTRY.values():
        for alias in spec.aliases:
            if alias in lowered:
                candidates.append((len(alias), spec.id))
    if not candidates:
        return None
    return max(candidates)[1]


def canonical_label(text: str) -> Optional[str]:
    """Free text to the canonical display label, or None."""
    method_id = canonical_id(text)
    return REGISTRY[method_id].label if method_id else None


def for_aquifer(aquifer: str, implemented_only: bool = True) -> list:
    return [s for s in all_specs(implemented_only) if s.supports_aquifer(aquifer)]


def for_test_type(test_type: str, implemented_only: bool = True) -> list:
    return [s for s in all_specs(implemented_only) if s.supports_test_type(test_type)]


def supported_aquifers(implemented_only: bool = True) -> set:
    """Aquifer types with at least one implemented method.

    This is what tells the chat layer honestly that unconfined has no calculator
    yet, instead of that fact being hardcoded in method_selection.py.
    """
    out = set()
    for spec in all_specs(implemented_only):
        out |= set(spec.aquifer_types)
    return out


def build(id_or_label: str):
    """Instantiate the calculator for a method id or label."""
    return require(id_or_label).build()
