"""Method selection — turns conversation context into a concrete method suggestion.

This is the bridge between the Consultation tab and the Calculator Workspace.
The Manager agent classifies intent with an LLM; this module then decides which
of the implemented calculators the user should actually run, and why.

The logic is deliberately deterministic and LLM-free so it can be unit tested
without network access, and so the reason shown to the user is always traceable
to a stated rule rather than to model output.

Selection rules, grounded in the source papers:

  * An explicitly named method always wins — the user is the expert on their own
    test setup.
  * Large-diameter well (dug well, caisson, or a stated radius above
    LARGE_WELL_RADIUS_M) with drawdown measured in the pumped well
    → Papadopulos-Cooper (1967). Theis assumption (5) is a well of
    infinitesimal diameter; a wide well stores enough water to dominate early
    drawdown.
  * An explicit late-time / straight-line / semi-log framing
    → Cooper-Jacob (1946), valid where u < 0.02 (paper p. 3).
  * A leaky aquifer → Hantush-Jacob (1955), which is the only method registered
    for that type. This arm is registry-driven, so it extends to unconfined and
    fractured automatically once those calculators exist.
  * Otherwise, confined aquifer → Theis (1935), the general full-curve method.
  * Aquifer types with no implemented calculator say so explicitly rather than
    silently offering Theis.
"""

import logging
import re
from typing import Optional

from src.calculator import registry
from src.calculator.parameters import summarise_parameters

logger = logging.getLogger(__name__)

# ── Canonical method labels ───────────────────────────────────────────────────
# Derived from the registry rather than duplicated, so a new calculator becomes
# selectable here the moment it is registered. These names are kept as module
# constants because they read far better at the call sites below than
# registry.get("theis_1935").label would.
THEIS = registry.REGISTRY["theis_1935"].label
COOPER_JACOB = registry.REGISTRY["cooper_jacob_1946"].label
PAPADOPULOS_COOPER = registry.REGISTRY["papadopulos_cooper_1967"].label
HANTUSH_JACOB = registry.REGISTRY["hantush_jacob_1955"].label

SUPPORTED_METHODS = registry.labels()

# Aquifer types with an implemented calculator. Derived, so when the first leaky
# method lands, rule 2 below stops declaring leaky unsupported on its own.
SUPPORTED_AQUIFERS = registry.supported_aquifers()

# Theis (1935) assumption (5) is a well of infinitesimal diameter. A casing
# radius beyond this makes wellbore storage non-negligible; DataValidator uses
# the same threshold for its Papadopulos-Cooper recommendation.
LARGE_WELL_RADIUS_M = 0.15

# ── Phrase cues ───────────────────────────────────────────────────────────────
_LARGE_WELL_CUES = re.compile(
    r"(large[- ]diameter|large diameter|dug well|open well|caisson|"
    r"wellbore storage|well ?bore storage|borehole storage|"
    r"pumped well itself|pumping well itself|in the pumped well)",
    re.IGNORECASE,
)
_LATE_TIME_CUES = re.compile(
    r"(late[- ]time|straight[- ]line|semi[- ]?log|log cycle|jacob)",
    re.IGNORECASE,
)

# "0.6 m diameter", "diameter of 1.2 m", "radius 0.8 metres"
_DIMENSION = re.compile(
    r"(?:(?P<lead_kind>diameter|radius)\s*(?:of|=|:)?\s*(?P<lead_val>\d+(?:\.\d+)?)\s*"
    r"(?P<lead_unit>m\b|metre|meter|cm\b|mm\b)"
    r"|(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>m\b|metre|meter|cm\b|mm\b)\s*"
    r"(?P<kind>diameter|radius))",
    re.IGNORECASE,
)

_TO_METRES = {"m": 1.0, "metre": 1.0, "meter": 1.0, "cm": 0.01, "mm": 0.001}


def canonical_method(name: Optional[str]) -> Optional[str]:
    """Map a free-form method name onto a canonical label, or None.

    Delegates to the registry, which matches aliases longest-first so that the
    more specific "Papadopulos-Cooper" beats the "cooper" it shares with
    "Cooper-Jacob". Methods that are real but not implemented — Neuman,
    Boulton, Moench — resolve to None rather than being coerced onto a
    present one.
    """
    return registry.canonical_label(name)


def extract_well_radius_m(text: str) -> Optional[float]:
    """Largest well radius stated in the text, in metres, or None.

    A stated diameter is halved. Returns the largest match so that "a 2 m
    diameter well with a 0.1 m screen" is treated as the large-diameter case.
    """
    if not text:
        return None

    radii = []
    for m in _DIMENSION.finditer(text):
        kind = m.group("lead_kind") or m.group("kind")
        value = m.group("lead_val") or m.group("val")
        unit = m.group("lead_unit") or m.group("unit")
        if not (kind and value and unit):
            continue
        metres = float(value) * _TO_METRES.get(unit.lower().rstrip("."), 1.0)
        if kind.lower() == "diameter":
            metres /= 2.0
        radii.append(metres)

    return max(radii) if radii else None


# Words that mean the data are a recovery record rather than pumping drawdown.
_RECOVERY_CUES = re.compile(
    r"\brecovery\b|residual drawdown|"
    r"pump(?:ing)?\s+(?:was\s+)?(?:stopped|switched off|shut off|turned off)",
    re.IGNORECASE,
)


def describes_recovery(text: str) -> bool:
    """Whether a message describes a recovery record rather than pumping drawdown.

    Public because the manager needs it to decide whether a message is worth
    suggesting from at all: "the pump stopped after 24 hours and I kept reading
    water levels as they came back up" names no aquifer and no method, so
    without this the suggestion was never attempted and the workspace was never
    primed, however well the consultant explained recovery in prose.
    """
    return bool(text) and bool(_RECOVERY_CUES.search(text))


# Words that mean the readings are concentrations, and the ones that say the
# tracer was drawn to a pumped well rather than carried by ambient flow.
_TRACER_CUES = re.compile(
    r"\btracer\b|breakthrough curve|\bdye test\b|\bbromide\b|\bfluorescein\b|"
    r"\brhodamine\b|\buranine\b",
    re.IGNORECASE,
)
_FORCED_CUES = re.compile(
    r"\bconvergent\b|\bconverging\b|forced[- ]gradient|\bpump(?:ed|ing)?\b",
    re.IGNORECASE,
)


def describes_tracer(text: str) -> bool:
    """Whether a message describes a tracer test. Public for the same reason as
    `describes_recovery`: it may name no aquifer and no method."""
    return bool(text) and bool(_TRACER_CUES.search(text))


def suggest_method(
    aquifer_context: str = "unknown",
    selected_method: str = "",
    user_message: str = "",
    time_s=None,
    drawdown_m=None,
    Q: Optional[float] = None,
) -> dict:
    """Decide which calculator to recommend.

    Args:
        aquifer_context: confined | unconfined | leaky | fractured | unknown
        selected_method: method the Manager extracted from the message, if any
        user_message:    the raw user text, scanned for well-geometry cues
        time_s, drawdown_m: the uploaded series, when one has been staged. Given
                        these, the Bourdet derivative reads the flow regime
                        straight off the data, which outranks every keyword
                        heuristic below.
        Q:              pumping rate [m³/s], used only to turn a radial plateau
                        into an implied transmissivity for the explanation

    Returns a suggestion card:
        {
          "method":     canonical label, or "" when nothing is implemented,
          "reason":     one-sentence justification shown in the UI,
          "confidence": 0.0-1.0,
          "supported":  bool — False when the aquifer type has no calculator,
          "evidence":   optional, present when the data itself was consulted,
        }
    """
    text = user_message or ""

    # ── 1. An explicitly named method wins ────────────────────────────────────
    #
    # The user's own words are read first, and the classifier's extraction only
    # afterwards. Two reasons, both seen on the deployed site: the extraction can
    # be wrong — a question about Papadopulos-Cooper came back as Cooper-Jacob —
    # and `selected_method` persists across turns, so a method named three
    # messages ago would otherwise outrank the one named in this one.
    named = canonical_method(text) or canonical_method(selected_method)
    if named:
        return {
            "method": named,
            "reason": f"You named {named}, so that is what the workspace will run.",
            "confidence": 0.95,
            "supported": True,
        }

    # ── 1b. Recovery data → the recovery method ───────────────────────────────
    # Ahead of the data-driven diagnosis, which reads a pumping-phase derivative:
    # a recovery record, with drawdown falling, would be misread by it.
    if _RECOVERY_CUES.search(text):
        recovery = registry.for_test_type(registry.RECOVERY)
        if recovery:
            return {
                "method": recovery[0].label,
                "reason": (recovery[0].selection_rationale + " Choose the Recovery "
                           "test type in Step 2 and enter how long the well pumped."),
                "confidence": 0.85,
                "supported": True,
            }

    # ── 1c. A tracer test → a transport method ────────────────────────────────
    # Ahead of the diagnosis for the same reason: a derivative reads drawdown.
    if describes_tracer(text):
        method_id = "thiery_2022" if _FORCED_CUES.search(text) else "ogata_banks_1961"
        spec = registry.REGISTRY[method_id]
        return {
            "method": spec.label,
            "reason": (spec.selection_rationale + " Choose the Tracer test type in "
                       "Step 2 and enter the travel distance."),
            "confidence": 0.8,
            "supported": True,
        }

    # ── 2. The data outranks the prose ────────────────────────────────────────
    # Drawdown curves for different aquifer types all look like smooth rising
    # lines; their derivatives do not. When a series is available, reading the
    # flow regime from it beats guessing from the words in the user's message —
    # a user who says "dug well" may still have a dataset showing plain radial
    # flow, and vice versa.
    diagnosis = _diagnose_from_data(time_s, drawdown_m, Q)
    if diagnosis is not None:
        return diagnosis

    # ── 3. Unsupported aquifer types — say so, do not silently offer Theis ────
    # Both the check and the wording come from the registry, so this claim stays
    # true as methods are added instead of going stale.
    known_aquifers = set(registry.AQUIFER_TYPES)
    if aquifer_context in known_aquifers - SUPPORTED_AQUIFERS:
        covered = sorted(SUPPORTED_AQUIFERS)
        available = ", ".join(registry.labels())
        return {
            "method": "",
            "reason": (
                f"No calculator is implemented for {aquifer_context} aquifers yet — "
                f"the workspace currently covers {' and '.join(covered)} aquifers only "
                f"({available}). "
                "I can still explain the appropriate method and its assumptions."
            ),
            "confidence": 0.9,
            "supported": False,
        }

    # ── 4. Large-diameter well → Papadopulos-Cooper ───────────────────────────
    radius = extract_well_radius_m(text)
    if _LARGE_WELL_CUES.search(text) or (radius is not None
                                          and radius > LARGE_WELL_RADIUS_M):
        if radius is not None and radius > LARGE_WELL_RADIUS_M:
            why = (f"a well radius of {radius:.2f} m exceeds the "
                   f"{LARGE_WELL_RADIUS_M} m at which wellbore storage becomes "
                   "significant")
        else:
            why = "you described a large-diameter well or wellbore storage"
        return {
            "method": PAPADOPULOS_COOPER,
            "reason": (
                f"Papadopulos-Cooper (1967) because {why}. Theis assumption (5) "
                "is a well of infinitesimal diameter, which does not hold here. "
                "Note the paper's caveat (p. 244): T is reliable, S only to an "
                "order of magnitude."
            ),
            "confidence": 0.85,
            "supported": True,
        }

    # ── 5. Late-time / straight-line framing → Cooper-Jacob ───────────────────
    if _LATE_TIME_CUES.search(text):
        return {
            "method": COOPER_JACOB,
            "reason": (
                "Cooper-Jacob (1946) because you described late-time or "
                "straight-line analysis. It fits a line to s vs log t where "
                "u < 0.02 (paper p. 3); early-time points are excluded "
                "automatically."
            ),
            "confidence": 0.8,
            "supported": True,
        }

    # ── 6. A supported non-confined aquifer routes to its own method ──────────
    # Derived from the registry, so this arm starts working for unconfined and
    # fractured the moment a calculator for them is registered.
    if aquifer_context in SUPPORTED_AQUIFERS and aquifer_context != "confined":
        for spec in registry.for_aquifer(aquifer_context):
            return {
                "method": spec.label,
                "reason": spec.selection_rationale,
                "confidence": 0.8,
                "supported": True,
            }

    # ── 7. Default for a confined aquifer ─────────────────────────────────────
    confident = aquifer_context == "confined"
    return {
        "method": THEIS,
        "reason": (
            "Theis (1935) is the general full-curve method for a confined "
            "aquifer and uses every data point, so it is the safe default. "
            + ("" if confident else
               "Aquifer type was not stated — confirm the aquifer is confined "
               "before relying on the result.")
        ),
        "confidence": 0.75 if confident else 0.5,
        "supported": True,
    }


# ── Data-driven diagnosis ─────────────────────────────────────────────────────

# Two thresholds, because the two outcomes carry different stakes.
#
# Recommending a method overrides what the user told us, so it needs real
# confidence. Warning that the data contradicts every method we have is a much
# cheaper claim — it commits us to nothing except caution — so a weaker reading
# is still worth surfacing. Using one threshold for both made the warning branch
# unreachable, since a barrier signature is deliberately reported at lower
# confidence than a radial one.
_RECOMMEND_CONFIDENCE_FLOOR = 0.7
_WARN_CONFIDENCE_FLOOR = 0.5


def _diagnose_from_data(time_s, drawdown_m, Q=None) -> Optional[dict]:
    """Turn a Bourdet derivative reading into a suggestion card, or None.

    Returns None whenever the data cannot support a confident call — too few
    points, an ambiguous shape, or a regime whose method is not implemented —
    so the caller falls through to the keyword rules rather than guessing.
    """
    if time_s is None or drawdown_m is None:
        return None

    try:
        from src.calculator.derivative import diagnose
        signature = diagnose(time_s, drawdown_m, Q=Q)
    except (ValueError, TypeError) as exc:
        # A malformed series must never break method selection.
        logger.debug("Derivative diagnosis unavailable: %s", exc)
        return None

    implemented = [registry.get(mid) for mid in signature.suggests]
    implemented = [spec for spec in implemented if spec is not None]

    if not implemented:
        # The derivative recognised something real — a barrier, linear flow —
        # that has no calculator. Say so rather than silently falling through to
        # a method whose assumptions the data has just contradicted.
        if (signature.regime in ("barrier_or_boundary", "linear_flow")
                and signature.confidence >= _WARN_CONFIDENCE_FLOOR):
            return {
                "method": "",
                "reason": (
                    f"{signature.evidence} No method for this behaviour is "
                    "implemented yet, so any fit would rest on assumptions your "
                    "data contradicts."
                ),
                "confidence": signature.confidence,
                "supported": False,
                "evidence": signature.evidence,
            }
        return None

    if signature.confidence < _RECOMMEND_CONFIDENCE_FLOOR:
        return None

    spec = implemented[0]
    return {
        "method": spec.label,
        "reason": (
            f"{signature.evidence} That points to {spec.label}. "
            "This reading comes from the shape of your own data rather than "
            "from how the test was described."
        ),
        "confidence": signature.confidence,
        "supported": True,
        "evidence": signature.evidence,
    }


def format_suggestion(suggestion: dict) -> str:
    """Render a suggestion card as chat markdown."""
    if not suggestion:
        return ""
    if not suggestion.get("supported", True) or not suggestion.get("method"):
        return f"**Method selection**\n\n{suggestion.get('reason', '')}"
    return (
        f"**Suggested method: {suggestion['method']}**\n\n"
        f"{suggestion['reason']}\n\n"
        f"*Confidence: {suggestion.get('confidence', 0):.0%}*"
    )


def _reports_S(result: dict) -> bool:
    """Whether the method estimated S. A recovery analysis gives only S/S', so
    printing its S field (0.0) would state a storativity nobody measured."""
    params = result.get("parameters")
    return not params or any(p.get("name") == "S" for p in params)


def summarise_result(result: dict, calc_input: dict = None) -> str:
    """Render a completed calculation as chat markdown.

    `result` is the plain-dict form of CalculationResult that the workspace
    writes into AgentState (numpy arrays stripped out).
    """
    if not result:
        return ""
    if not result.get("success", False):
        return (
            "The last analysis did not converge: "
            f"{result.get('error_message', 'unknown error')}"
        )

    if result.get("response_unit"):
        lines = [f"**{result.get('method', 'Analysis')} — completed**", ""]
        lines += [f"- {line}" for line in summarise_parameters(result.get("parameters"))]
        lines.append(f"- **R²** = {result.get('r_squared', 0):.4f}, RMSE = "
                     f"{result.get('rmse', 0):.3g} {result['response_unit']}")
        notes = result.get("validity_notes", [])
        if notes:
            lines += ["", "**Validity notes:**"] + [f"- {n}" for n in notes[:4]]
        return "\n".join(lines)

    lines = [
        f"**{result.get('method', 'Analysis')} — completed**",
        "",
        f"- **T** = {result.get('T_day', 0):.2f} m²/day "
        f"({result.get('T', 0):.3e} m²/s)",
        (f"- **S** = {result.get('S', 0):.2e}" if _reports_S(result)
         else "- **S** — not determined by this method"),
        f"- **R²** = {result.get('r_squared', 0):.4f}, "
        f"RMSE = {result.get('rmse', 0) * 1000:.2f} mm",
    ]

    if calc_input:
        lines.append(
            f"- Test setup: Q = {calc_input.get('Q', 0) * 86400:.1f} m³/day, "
            f"r = {calc_input.get('r', 0):.1f} m, "
            f"{calc_input.get('n_points', 0)} points"
        )

    notes = result.get("validity_notes", [])
    if notes:
        lines += ["", "**Validity notes:**"]
        lines += [f"- {n}" for n in notes[:4]]
        if len(notes) > 4:
            lines.append(f"- …and {len(notes) - 4} more in the workspace.")

    return "\n".join(lines)
