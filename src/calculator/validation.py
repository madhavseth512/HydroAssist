"""Input checks shared by every calculator.

`_validate` is currently duplicated almost verbatim in theis.py, cooper_jacob.py
and papadopulos_cooper.py — three copies of the same five checks, which drift the
moment one is edited. This is the one copy.

The error strings are reproduced exactly as they appear today, because several
existing tests assert on them with `pytest.raises(match=...)`, and because they
are user-facing: each says what is wrong and what to do about it.
"""

import logging
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Below this, a nonlinear fit of two or more parameters is not meaningfully
# constrained. DataValidator uses the same threshold.
MIN_DATA_POINTS = 10

# An observation distance under this is far more likely to be the pumping well's
# own casing radius entered by mistake than a real observation well.
SUSPICIOUS_RADIUS_M = 0.5


def validate_series(inputs, *, min_points: int = MIN_DATA_POINTS,
                    require_rate: bool = True,
                    require_distance: bool = True) -> Optional[str]:
    """Return a user-facing error string, or None when the inputs are usable.

    `require_rate` and `require_distance` exist for test types that genuinely
    have neither: a slug test has no pumping rate at all, and its driving term is
    an initial head displacement.
    """
    if require_rate:
        if inputs.Q is None:
            return "Pumping rate Q is required for this test type."
        if inputs.Q <= 0:
            return (f"Pumping rate Q must be positive (got {inputs.Q}). "
                    "Ensure Q is in m³/s.")

    if require_distance:
        if inputs.r <= 0:
            return f"Observation distance r must be positive (got {inputs.r} m)."
        if inputs.r < SUSPICIOUS_RADIUS_M:
            # A warning, not an error: piezometers close to the pumped well are
            # unusual but legitimate.
            logger.warning(
                f"r = {inputs.r}m is very small — verify this is the observation well "
                "distance (typically 10–200m), not the pumping well casing radius."
            )

    t, s = inputs.get_series()

    if len(t) < min_points:
        return f"Insufficient data: {len(t)} points (minimum {min_points} required)."
    if np.any(s < 0):
        return ("Negative drawdown values detected — remove recovery phase data "
                "before fitting.")
    if np.any(t <= 0):
        return ("Non-positive time values detected — t=0 should be removed by "
                "the formatter.")
    return None


def resolve_path(obj, path: str):
    """Follow a dotted attribute path such as "well.r_w", returning None if any
    step is missing or None. Indexing like "aquitards[0].b_prime" is supported so
    a method can require the first confining layer."""
    current = obj
    for part in path.split("."):
        index = None
        if part.endswith("]") and "[" in part:
            part, _, raw = part.partition("[")
            index = int(raw.rstrip("]"))
        current = getattr(current, part, None)
        if current is None:
            return None
        if index is not None:
            try:
                current = current[index]
            except (IndexError, KeyError, TypeError):
                return None
    return current


def require_geometry(inputs, paths: Sequence[str],
                     method_label: str = "This method") -> Optional[str]:
    """Check that every geometry input a method needs has been supplied.

    Driven by the `requires` tuple on the method's registry entry, so a method
    declares its own needs in one place and gets a usable error message for free
    instead of failing somewhere inside the objective function.
    """
    missing = [p for p in paths if resolve_path(inputs, p) is None]
    if not missing:
        return None

    friendly = ", ".join(_FRIENDLY_NAMES.get(p, p) for p in missing)
    return (
        f"{method_label} needs {friendly}, which "
        f"{'was' if len(missing) == 1 else 'were'} not supplied. "
        "Enter the missing well or aquifer geometry in Step 2."
    )


# Dotted paths read like code; these read like hydrogeology.
_FRIENDLY_NAMES = {
    "well.r_w": "the well screen radius r_w",
    "well.r_c": "the casing radius r_c",
    "well.screen_top_m": "the screen top depth",
    "well.screen_bottom_m": "the screen bottom depth",
    "aquifer.b": "the saturated aquifer thickness b",
    "aquifer.Kz_over_Kr": "the vertical anisotropy ratio Kz/Kr",
    "observation.r": "the observation well distance r",
    "observation.z_m": "the piezometer depth",
    "aquitards[0].b_prime": "the aquitard thickness b'",
    "aquitards[0].K_prime": "the aquitard vertical conductivity K'",
    "aquitards[0].Ss_prime": "the aquitard specific storage S's",
    "fracture.block_half_width": "the fracture block half-width",
    "slug.H0": "the initial head displacement H0",
}
