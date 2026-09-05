"""Laplace-domain well-test solutions, factored into a shared kernel plus roots.

The observation that makes the domain increment cheap: almost every radial-flow
solution in the catalogue is the *same* Laplace-domain kernel

    s̄(p) = Q · K₀(q·r) / (2π · T · p)

with the Theis characteristic root  q = √(pS/T)  replaced by a method-specific
q(p), optionally wrapped in the van Everdingen-Hurst inner boundary condition
that accounts for wellbore storage and skin.

    Theis            q = √(pS/T)
    Hantush-Jacob    q = √(pS/T + 1/B²)          — leakage adds a constant
    Boulton          q = √((p/T)(S + Sy·α/(p+α))) — delayed yield adds a pole
    Warren-Root      q = √(p·f(p)·S_tot/T)        — double porosity modifies p

So a new leaky or dual-porosity method is a few lines here plus a parameter spec,
rather than a new module. Papadopulos-Cooper is already a special case: the
wellbore-storage kernel evaluated at r = r_w with the Theis root and no skin.

Numerical note on the Bessel functions, which differs between the two kernels:

  * `with_wellbore_storage` uses K₀ and K₁ only inside a ratio at one argument,
    so the exp(-x) factor cancels exactly. It uses the exponentially scaled
    `kve`, which stays finite where plain `kv` underflows to zero and turns the
    ratio into 0/0.
  * `radial_line_source` has K₀ alone, where the exponential decay is the
    physics — drawdown has not yet reached that radius. Plain `kv` is correct
    there and its underflow to zero is the right answer, not a bug.
"""

import numpy as np
from scipy.special import kv, kve

# Below this the modified Bessel functions are evaluated by their small-argument
# series instead, where kv/kve lose precision or return inf.
_SMALL = 1e-8

# Euler-Mascheroni constant, from the K₀ small-argument expansion.
_GAMMA = 0.5772156649015329


def _safe_positive(x):
    """Clamp to a strictly positive floor so log and division stay finite."""
    return np.maximum(np.asarray(x, dtype=float), 1e-300)


# ── Characteristic roots ──────────────────────────────────────────────────────
# Each returns q(p) with units of 1/length. Adding a method usually means adding
# one of these.

def q_theis(p, T: float, S: float):
    """Non-leaky confined aquifer. Theis (1935)."""
    return np.sqrt(np.asarray(p, dtype=float) * S / T)


def q_hantush_jacob(p, T: float, S: float, B: float):
    """Leaky confined aquifer with no aquitard storage. Hantush & Jacob (1955).

    B = √(T·b'/K') is the leakage factor. As B → ∞ leakage vanishes and this
    reduces exactly to the Theis root, which is the primary correctness test for
    the method.
    """
    return np.sqrt(np.asarray(p, dtype=float) * S / T + 1.0 / (B * B))


def q_boulton(p, T: float, S: float, Sy: float, alpha: float):
    """Unconfined aquifer with delayed yield. Boulton (1963).

    The empirical drainage constant alpha adds a pole: at early time the aquifer
    responds with storativity S, at late time with specific yield Sy, and alpha
    sets the transition. As alpha → ∞ drainage is instantaneous and this tends to
    Theis with S + Sy.
    """
    p = np.asarray(p, dtype=float)
    storage = S + Sy * alpha / (p + alpha)
    return np.sqrt(p * storage / T)


def q_warren_root(p, T: float, S_total: float, omega: float, lam: float):
    """Double-porosity fractured aquifer. Warren & Root (1963).

    omega is the fraction of storage in the fractures and lambda the
    interporosity flow coefficient. omega → 1 puts all storage in the fractures
    and recovers Theis.
    """
    p = np.asarray(p, dtype=float)
    # The classic transfer function: p is replaced by p·f(p).
    f = (omega * (1.0 - omega) * p + lam) / ((1.0 - omega) * p + lam)
    return np.sqrt(p * f * S_total / T)


def q_hantush_1960(p, T: float, S: float, b_prime: float, K_prime: float,
                   Ss_prime: float):
    """Leaky aquifer WITH storage in the aquitard. Hantush (1960).

    The aquitard no longer transmits leakage instantaneously; its own storage
    delays the response, which the coth term expresses. As Ss' → 0 the term
    collapses to the constant 1/B² of Hantush-Jacob.
    """
    p = np.asarray(p, dtype=float)
    # Argument of the aquitard diffusion term.
    arg = np.sqrt(p * Ss_prime * b_prime * b_prime / K_prime)
    arg = _safe_positive(arg)
    # coth(x) = 1/tanh(x); tanh saturates to 1 for large x, which is the
    # no-storage limit, so this stays well behaved at both ends.
    coth = 1.0 / np.tanh(arg)
    leak = (K_prime / (T * b_prime)) * arg * coth
    return np.sqrt(p * S / T + leak)


# ── Kernels ───────────────────────────────────────────────────────────────────

def _k0(x):
    """K₀ with a small-argument series where kv loses precision."""
    x = _safe_positive(x)
    small = x < _SMALL
    if not np.any(small):
        return kv(0, x)
    out = np.where(small, -np.log(x / 2.0) - _GAMMA, 0.0)
    big = ~small
    if np.any(big):
        out = np.asarray(out, dtype=float)
        out[big] = kv(0, x[big])
    return out


def radial_line_source(p, Q: float, T: float, r: float, q):
    """Drawdown transform for a line source of infinitesimal diameter.

        s̄(p) = Q · K₀(q·r) / (2π · T · p)

    Plain K₀ is correct here: its exponential decay at large q·r is the physical
    statement that drawdown has not yet propagated to that radius, so underflow
    to zero is the right answer.
    """
    p = np.asarray(p, dtype=float)
    return Q * _k0(np.asarray(q, dtype=float) * r) / (2.0 * np.pi * T * p)


def with_wellbore_storage(p, Q: float, T: float, r: float,
                          r_w: float, r_c: float, q, skin: float = 0.0):
    """Drawdown transform under the van Everdingen-Hurst inner boundary condition.

    Accounts for water released from storage within the well casing itself, which
    dominates early-time drawdown in a large-diameter well and which the line
    source of `radial_line_source` ignores.

    At r = r_w with the Theis root and skin = 0 this is exactly the
    Papadopulos-Cooper (1967) equation 10:

        s̄_w(p) = Q·K₀(q·r_w) /
                 {π·p·[r_c²·p·K₀(q·r_w) + 2·r_w·T·q·K₁(q·r_w)]}

    K₀ and K₁ appear only in a ratio at the same argument, so the exponentially
    scaled forms substitute exactly and keep the ratio finite where the plain
    functions would both underflow to zero.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    x_w = _safe_positive(q * r_w)
    K0w = kve(0, x_w)
    K1w = kve(1, x_w)

    if r == r_w:
        K0r = K0w
    else:
        # Scaled consistently with the denominator, at this radius' own argument.
        x_r = _safe_positive(q * r)
        K0r = kve(0, x_r) * np.exp(-(x_r - x_w))

    skin_term = 2.0 * r_w * T * q * K1w * (1.0 + skin * x_w) if skin else \
        2.0 * r_w * T * q * K1w

    denom = np.pi * p * (r_c * r_c * p * K0w + skin_term)
    return np.where(np.abs(denom) < 1e-300, 0.0, Q * K0r / np.where(
        np.abs(denom) < 1e-300, 1.0, denom))


# ── Convenience: complete transforms ready for inversion ──────────────────────

def theis_transform(Q: float, T: float, S: float, r: float):
    """F̄(p) for Theis, ready to hand to stehfest_invert_array."""
    return lambda p: radial_line_source(p, Q, T, r, q_theis(p, T, S))


def hantush_jacob_transform(Q: float, T: float, S: float, r: float, B: float):
    """F̄(p) for Hantush-Jacob (1955)."""
    return lambda p: radial_line_source(p, Q, T, r, q_hantush_jacob(p, T, S, B))


def papadopulos_cooper_transform(Q: float, T: float, S: float,
                                 r_w: float, r_c: float):
    """F̄(p) for Papadopulos-Cooper (1967), evaluated at the well face."""
    return lambda p: with_wellbore_storage(
        p, Q, T, r_w, r_w, r_c, q_theis(p, T, S)
    )
