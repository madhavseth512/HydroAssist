"""Bourdet log-derivative and flow-regime diagnosis.

Reference: Bourdet, D., Ayoub, J.A. & Pirard, Y.M. (1989). Use of pressure
derivative in well test interpretation. SPE Formation Evaluation, 4(2), 293-302.

Modern well-test practice runs the derivative *first* and picks the model from
its shape, rather than fitting a model chosen in advance and checking R^2
afterwards. The reason is that drawdown curves for different aquifer types look
much alike on a log-log plot — every one of them is some smooth rising curve —
while their derivatives look completely different:

    ds/d ln t  flat                  radial flow          -> Theis
               unit slope, early     wellbore storage     -> Papadopulos-Cooper
               dips and stays low    leakage              -> Hantush-Jacob
               dips then recovers    double porosity      -> Warren-Root, Moench
               doubles at late time  no-flow boundary     -> image well needed
               half slope            linear flow          -> Gringarten fracture
               two plateaus, dip     delayed yield        -> Neuman, Boulton

That is why this module matters beyond parity with AQTESOLV, which has the plot
but does not connect it to any advice: it turns `suggest_method` from keyword
matching on the user's sentence into a reading of their actual data.

The derivative is computed on ln(t) rather than t because the flat value then has
a direct physical meaning. At late time Theis gives

    s = (Q/4piT) * (ln(4Tt/r^2 S) - gamma)      so      ds/d ln t = Q/(4 pi T)

so the height of the flat portion IS the transmissivity, readable straight off the
plot. That identity is the module's primary test.

Differentiating measured data amplifies noise, which is why Bourdet's scheme uses
neighbours separated by a smoothing window L in log space rather than adjacent
points, and takes a weighted average of the left and right slopes.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Smoothing window in natural-log units. Bourdet et al. recommend 0.1-0.3; 0.15
# is the usual starting point, wide enough to suppress logger noise and narrow
# enough to keep a genuine transition visible.
DEFAULT_WINDOW = 0.15

# Fewer points than this cannot support a differentiated curve at all.
MIN_POINTS = 5

# How close a measured log-log slope must be to a textbook value to be called it.
SLOPE_TOLERANCE = 0.15


@dataclass
class DerivativeSignature:
    """What the derivative's shape implies about the aquifer.

    Deliberately reports evidence rather than a verdict: field data is noisy, and
    the honest output is "the late-time derivative is falling, which is
    consistent with leakage or a recharge boundary", not "this is a leaky
    aquifer". `suggests` names method ids the registry knows.
    """

    regime: str                       # radial | wellbore_storage | leakage | ...
    confidence: float                 # 0-1
    evidence: str                     # one sentence, shown to the user
    suggests: tuple = ()              # registry method ids, best first
    early_slope: Optional[float] = None   # d ln(D) / d ln(t), early time
    late_slope: Optional[float] = None    # ... late time
    flat_value: Optional[float] = None    # derivative height on a radial plateau
    implied_T: Optional[float] = None     # Q / (4 pi * flat_value)
    notes: list = field(default_factory=list)


def log_derivative(time_s, drawdown_m, window: float = DEFAULT_WINDOW):
    """Bourdet log-derivative ds/d ln(t).

    For each point, neighbours are chosen at least `window` away in ln(t) — not
    the adjacent samples, which would amplify measurement noise — and the left
    and right slopes are combined as a weighted average, each weighted by the
    *opposite* span:

        D_i = (m_left * dl_right + m_right * dl_left) / (dl_left + dl_right)

    Returns an array the same length as the input. Points too close to either end
    to have a full window fall back to the widest neighbours available, so the
    curve is defined everywhere but is least reliable at the extremes.
    """
    t = np.asarray(time_s, dtype=float)
    s = np.asarray(drawdown_m, dtype=float)

    if t.size != s.size:
        raise ValueError("time and drawdown must have the same length.")
    if t.size < MIN_POINTS:
        raise ValueError(
            f"The derivative needs at least {MIN_POINTS} points, got {t.size}."
        )
    if np.any(t <= 0):
        raise ValueError("The log-derivative requires strictly positive times.")

    order = np.argsort(t)
    t, s = t[order], s[order]
    ln_t = np.log(t)
    n = t.size

    out = np.empty(n)
    for i in range(n):
        left = _neighbour(ln_t, i, window, direction=-1)
        right = _neighbour(ln_t, i, window, direction=+1)

        dl_left = ln_t[i] - ln_t[left]
        dl_right = ln_t[right] - ln_t[i]

        if dl_left <= 0 and dl_right <= 0:
            out[i] = 0.0
            continue
        if dl_left <= 0:                       # first point: right slope only
            out[i] = (s[right] - s[i]) / dl_right
            continue
        if dl_right <= 0:                      # last point: left slope only
            out[i] = (s[i] - s[left]) / dl_left
            continue

        m_left = (s[i] - s[left]) / dl_left
        m_right = (s[right] - s[i]) / dl_right
        out[i] = (m_left * dl_right + m_right * dl_left) / (dl_left + dl_right)

    # Restore the caller's ordering.
    restored = np.empty(n)
    restored[order] = out
    return restored


def _neighbour(ln_t: np.ndarray, i: int, window: float, direction: int) -> int:
    """Index of the nearest point at least `window` away in ln(t).

    Falls back to the outermost available point when the window cannot be
    satisfied, which happens within one window of each end of the record.
    """
    n = ln_t.size
    j = i
    while 0 <= j + direction < n:
        j += direction
        if abs(ln_t[j] - ln_t[i]) >= window:
            return j
    return max(0, min(n - 1, j))


def loglog_slope(x, y, mask=None) -> Optional[float]:
    """Least-squares slope of ln(y) against ln(x) — the slope seen on a log-log
    plot, which is what the textbook signatures are quoted in."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if mask is not None:
        x, y = x[mask], y[mask]

    valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return None
    slope, _ = np.polyfit(np.log(x[valid]), np.log(y[valid]), 1)
    return float(slope)


def diagnose(time_s, drawdown_m, Q: Optional[float] = None,
             window: float = DEFAULT_WINDOW) -> DerivativeSignature:
    """Read the derivative's shape and say which model it points to.

    `Q` is optional and used only to convert a radial plateau into an implied
    transmissivity, via ds/d ln t = Q/(4 pi T).
    """
    t = np.asarray(time_s, dtype=float)
    s = np.asarray(drawdown_m, dtype=float)

    # Checked before sorting: indexing one array by the other's argsort would
    # raise IndexError rather than the ValueError callers are prepared for.
    if t.size != s.size:
        raise ValueError(
            f"time and drawdown must have the same length, got {t.size} and {s.size}."
        )

    if t.size < MIN_POINTS:
        return DerivativeSignature(
            regime="insufficient_data", confidence=0.0,
            evidence=(f"Only {t.size} points; at least {MIN_POINTS} are needed "
                      "before the derivative says anything."),
        )

    order = np.argsort(t)
    t, s = t[order], s[order]
    d = log_derivative(t, s, window)

    n = t.size
    early = slice(0, max(3, n // 3))
    late = slice(max(0, n - max(3, n // 3)), n)

    early_slope = loglog_slope(t[early], d[early])
    late_slope = loglog_slope(t[late], d[late])

    # ── Wellbore storage: unit slope AND derivative lying on the drawdown ─────
    # During pure wellbore storage all the water comes from the casing, so
    # s = Qt/(pi r_c^2), which gives ds/d ln t = s exactly.
    if early_slope is not None and abs(early_slope - 1.0) < SLOPE_TOLERANCE:
        overlay = np.abs(d[early] - s[early]) / np.maximum(s[early], 1e-12)
        if float(np.median(overlay)) < 0.25:
            return DerivativeSignature(
                regime="wellbore_storage",
                confidence=0.85,
                evidence=(
                    f"The early-time derivative has unit slope "
                    f"({early_slope:.2f}) and lies on the drawdown curve, which "
                    "means early water is coming from storage in the well casing "
                    "rather than from the aquifer."
                ),
                suggests=("papadopulos_cooper_1967",),
                early_slope=early_slope, late_slope=late_slope,
            )

    # ── Late-time behaviour classifies the aquifer ────────────────────────────
    if late_slope is None:
        return DerivativeSignature(
            regime="indeterminate", confidence=0.2,
            evidence="The derivative is too scattered at late time to classify.",
            early_slope=early_slope, late_slope=late_slope,
        )

    flat_value = float(np.median(d[late]))
    implied_T = (Q / (4.0 * np.pi * flat_value)
                 if Q and flat_value > 0 else None)

    # ── Half slope: linear flow into a vertical fracture ─────────────────────
    # Checked BEFORE the generic rising-derivative branch below. A half slope is
    # a rising derivative, so testing "rising" first would shadow this entirely
    # and report every fractured well as a boundary effect — the less specific
    # and less useful of the two readings.
    whole_slope = loglog_slope(t, d)
    if whole_slope is not None and abs(whole_slope - 0.5) < SLOPE_TOLERANCE:
        return DerivativeSignature(
            regime="linear_flow",
            confidence=0.7,
            evidence=(
                f"The derivative follows a half slope ({whole_slope:.2f}), the "
                "signature of linear flow into a vertical fracture rather than "
                "radial flow to a well."
            ),
            suggests=(),
            early_slope=early_slope, late_slope=late_slope,
            notes=["Gringarten-Witherspoon (1972) is the matching model; it is "
                   "not implemented yet."],
        )

    # Falling derivative: pressure support arriving from somewhere.
    if late_slope < -0.30:
        return DerivativeSignature(
            regime="leakage_or_recharge",
            confidence=0.75,
            evidence=(
                f"The late-time derivative is falling (log-log slope "
                f"{late_slope:+.2f}), so drawdown is levelling off. Something is "
                "supplying water: leakage through a confining layer, a recharge "
                "boundary, or delayed yield from a water table."
            ),
            suggests=("hantush_jacob_1955",),
            early_slope=early_slope, late_slope=late_slope,
            notes=["A constant-head boundary produces the same signature; "
                   "the site's geology decides which."],
        )

    # Rising derivative: something is impeding flow.
    if late_slope > 0.30:
        return DerivativeSignature(
            regime="barrier_or_boundary",
            confidence=0.6,
            evidence=(
                f"The late-time derivative is rising (log-log slope "
                f"{late_slope:+.2f}), which suggests a no-flow boundary or a "
                "decrease in transmissivity away from the well. A derivative "
                "that doubles and flattens again is the classic barrier signature."
            ),
            suggests=(),
            early_slope=early_slope, late_slope=late_slope,
            notes=["No boundary model is implemented yet; T from an "
                   "infinite-acting fit will be biased."],
        )

    # ── Flat: infinite-acting radial flow ────────────────────────────────────
    evidence = (
        f"The late-time derivative is flat (log-log slope {late_slope:+.2f}), "
        "which is infinite-acting radial flow — the assumption Theis is built on."
    )
    if implied_T:
        evidence += (
            f" Its height gives T = Q/(4piD) = {implied_T * 86400:.1f} m²/day "
            "directly, independent of any curve fit."
        )

    return DerivativeSignature(
        regime="radial",
        confidence=0.8,
        evidence=evidence,
        suggests=("theis_1935", "cooper_jacob_1946"),
        early_slope=early_slope, late_slope=late_slope,
        flat_value=flat_value, implied_T=implied_T,
    )
