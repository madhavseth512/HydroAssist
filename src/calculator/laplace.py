"""Numerical inversion of Laplace-domain solutions.

Most of the well-test solutions still to be implemented — Hantush-Jacob, Hantush
(1960), Moench for leaky and for fractured aquifers, Boulton, Warren-Root — have
clean closed forms in the Laplace domain and awkward or oscillatory ones in the
time domain. Papadopulos-Cooper is already implemented this way: inverting the
analytical Laplace solution of the paper's equation 10 replaced direct integration
of the oscillatory equation 13, which gave 23-115% errors.

This module holds that machinery once, so each new method contributes only its
own transfer function.

**Why vectorisation is a prerequisite rather than an optimisation.** The original
implementation looped in Python: one call per time point per Stehfest node, so a
60-point curve cost 720 evaluations of the transform. A two-parameter fit runs
that a few hundred times, which is why Papadopulos-Cooper fitting takes 10-30
seconds. A four- to six-parameter Neuman or Moench fit explores far more of the
parameter space and would run on the order of 10^5 evaluations. Evaluating the
transform once on a flattened (n_times x N) grid instead of in nested Python loops
is what makes those methods usable at all.

**Accuracy.** Stehfest requires a smooth, non-oscillatory transform. It is
therefore right for every solution here and wrong for two things worth stating
explicitly:

  * Multi-rate schedules. Superpose single-rate responses in the TIME domain
    instead; a Laplace-domain step function is not smooth.
  * Underdamped (oscillatory) slug-test response, which needs de Hoog or similar.

N = 12 is the double-precision optimum: Stehfest needs roughly 2.2N significant
digits internally, so beyond about N = 18 round-off amplification dominates and
accuracy gets worse, not better.

**Measured accuracy envelope at N = 12.** The algorithm delivers roughly constant
ABSOLUTE accuracy, on the order of 1e-4 of the function's peak magnitude, rather
than constant relative accuracy. Measured on known pairs:

    f(t) = t^2      relative error 4.5e-5, flat across six log cycles of t
    f(t) = 1        relative error < 1e-6
    f(t) = e^-at    relative error 2e-2 once the value falls to ~1e-2,
                    and useless once it falls below ~1e-5

The practical consequence, worth knowing before choosing this inverter for a new
method: **it is excellent for functions that grow or plateau, and poor for
functions that decay toward zero.**

Drawdown grows logarithmically, so every pumping-test solution here sits in the
favourable regime — which is why the Theis kernel reproduces the closed form to
1e-4 across five log cycles. Slug-test head recovery, by contrast, decays
exponentially toward zero, so Stehfest is the wrong tool for it and the slug
methods will need de Hoog-Knight-Stokes or a direct time-domain solution. That is
a second, independent reason slug tests are a structural rather than a catalogue
addition.
"""

import logging
from functools import lru_cache
from math import factorial
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# Industry-standard order for smooth transforms in double precision.
DEFAULT_N = 12

# Above this, round-off amplification outweighs the extra terms.
MAX_RELIABLE_N = 18

LN2 = float(np.log(2.0))


@lru_cache(maxsize=8)
def stehfest_coefficients(N: int = DEFAULT_N) -> tuple:
    """The Stehfest (1970) V coefficients for an N-term inversion.

    Cached because they depend only on N, and returned as a tuple so the cache
    cannot hand out a mutable array that a caller might modify in place.
    """
    if N % 2 != 0:
        raise ValueError(f"Stehfest order N must be even, got {N}.")
    if N < 2:
        raise ValueError(f"Stehfest order N must be at least 2, got {N}.")
    if N > MAX_RELIABLE_N:
        logger.warning(
            f"Stehfest N={N} exceeds {MAX_RELIABLE_N}; round-off amplification "
            "usually makes large N less accurate, not more."
        )

    M = N // 2
    V = []
    for i in range(1, N + 1):
        total = 0.0
        for k in range(int(np.floor((i + 1) / 2)), min(i, M) + 1):
            total += (k ** M * factorial(2 * k) /
                      (factorial(M - k) * factorial(k) * factorial(k - 1) *
                       factorial(i - k) * factorial(2 * k - i)))
        V.append(((-1) ** (M + i)) * total)
    return tuple(V)


def stehfest_nodes(t: np.ndarray, N: int = DEFAULT_N) -> tuple:
    """The Laplace variables at which the transform must be evaluated.

    Returns (p, ln2_over_t) where p has shape (len(t), N). Exposed separately so
    a caller can inspect the sampling, and because building it is the only part
    that depends on t.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if np.any(t <= 0):
        raise ValueError("Laplace inversion requires strictly positive times.")

    ln2_over_t = LN2 / t                                   # (n_times,)
    i = np.arange(1, N + 1, dtype=float)                   # (N,)
    return ln2_over_t[:, None] * i[None, :], ln2_over_t


def stehfest_invert_array(f_bar: Callable, t, N: int = DEFAULT_N) -> np.ndarray:
    """Invert a Laplace transform at an array of times.

    `f_bar` must accept an ndarray of Laplace variables and return an ndarray of
    the same shape. It is called exactly once, on the flattened (n_times x N)
    grid — that single call is the whole point of this function.

        f(t) ≈ (ln2/t) · Σᵢ Vᵢ · F̄(i · ln2/t)
    """
    scalar_input = np.ndim(t) == 0
    p, ln2_over_t = stehfest_nodes(t, N)
    V = np.asarray(stehfest_coefficients(N), dtype=float)

    values = np.asarray(f_bar(p.ravel()), dtype=float).reshape(p.shape)
    out = ln2_over_t * (values @ V)

    return float(out[0]) if scalar_input else out


def stehfest_invert(f_bar: Callable, t: float, N: int = DEFAULT_N) -> float:
    """Invert at a single time. `f_bar` may be scalar- or array-valued.

    Prefer stehfest_invert_array for a whole curve; this exists for one-off
    evaluations and for transforms that are only defined scalar-wise.
    """
    V = stehfest_coefficients(N)
    ln2_over_t = LN2 / float(t)
    total = 0.0
    for i in range(1, N + 1):
        total += V[i - 1] * float(f_bar(i * ln2_over_t))
    return ln2_over_t * total


def cross_check_orders(f_bar: Callable, t, orders=(8, 12, 16),
                       rtol: float = 1e-3) -> dict:
    """Compare inversions at several Stehfest orders.

    Agreement across N is the standard cheap detector of a transform that
    violates the smoothness assumption, or of an algebra error in the transform
    itself: a correct, smooth F̄ gives nearly the same answer at N = 8, 12 and 16,
    while a wrong or non-smooth one diverges wildly.

    The default tolerance is deliberately loose. This is gross-error detection,
    not a precision measurement — N = 8 is genuinely coarser than N = 12 (about
    1.5e-4 relative on a power law), while a non-smooth transform disagrees by
    order unity. Tightening rtol toward the N = 8 noise floor would produce false
    alarms without catching anything extra.
    """
    results = {N: np.atleast_1d(stehfest_invert_array(f_bar, t, N)) for N in orders}
    stacked = np.vstack([results[N] for N in orders])

    reference = np.abs(stacked).max(axis=0)
    safe = np.where(reference > 0, reference, 1.0)
    spread = float(np.max((stacked.max(axis=0) - stacked.min(axis=0)) / safe))

    return {"results": results, "max_relative_spread": spread,
            "agrees": spread <= rtol}
