"""Power-law gate: is a "power-law / scale-free" claim actually supported?

Heavy-tailed data are routinely declared power-law on the strength of a straight line on
a log-log plot, which is not evidence. The Clauset-Shalizi-Newman procedure does it
properly: fit the tail by maximum likelihood (choosing the cutoff that minimises the
Kolmogorov-Smirnov distance), test goodness-of-fit with a seeded semiparametric
bootstrap, and, crucially, compare the power law against a lognormal and an exponential
with a likelihood-ratio test, because heavy tails are usually consistent with a
lognormal too. A power law is asserted only when it is not ruled out and no alternative
fits better; the common honest outcome is "cannot be distinguished from a lognormal".
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._numerics import _as_float
from ._report import Check, VerificationReport

#: Seeded-LCG constants for the deterministic bootstrap.
_LCG_A = 6364136223846793005
_LCG_C = 1442695040888963407
_MASK = (1 << 64) - 1


def verify_powerlaw(
    rows: Sequence[dict[str, Any]], column: str, *, reps: int = 300
) -> VerificationReport:
    """Verify that ``column`` plausibly follows a power law (Clauset et al. 2009).

    Fits the power-law tail, runs the bootstrap goodness-of-fit test, and compares it
    with a lognormal and an exponential. The verdict is ``unsound`` (the power law is
    ruled out by the bootstrap, or a lognormal/exponential fits significantly better),
    ``sound`` (a power law is plausible and no alternative is favoured: with a caveat
    when it merely cannot be distinguished from a lognormal), or ``inconclusive`` (too
    few values, or non-positive data).
    """
    xs = sorted(
        v for row in rows if (v := _as_float(row.get(column))) is not None and v > 0
    )
    if len(xs) < 50:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("a power-law test needs at least 50 positive values",),
        )
    xmin, alpha, d_obs, tail = _fit(xs)
    if len(tail) < 15:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("too few points in the fitted tail to test a power law",),
        )
    p_gof = _bootstrap_gof(xs, xmin, alpha, d_obs, reps)
    detail = f"alpha = {alpha:.2f}, xmin = {xmin:.3g}, bootstrap p = {p_gof:.2f}"
    if p_gof < 0.1:
        return VerificationReport(
            "unsound",
            alpha,
            True,
            (Check("power-law-fit", False, detail),),
            f"a power law is ruled out (bootstrap goodness-of-fit p = {p_gof:.2f})",
            (),
        )
    z_ln, p_ln = _vuong(tail, xmin, alpha, _lognormal_loglik(tail, xmin, xs))
    z_exp, p_exp = _vuong(tail, xmin, alpha, _exponential_loglik(tail, xmin))
    # an alternative significantly favoured (z < 0, p < 0.1) refutes the power law
    if (z_ln < 0 and p_ln < 0.1) or (z_exp < 0 and p_exp < 0.1):
        which = "a lognormal" if (z_ln < 0 and p_ln < 0.1) else "an exponential"
        return VerificationReport(
            "unsound",
            alpha,
            True,
            (Check("vs-alternatives", False, detail),),
            f"{which} fits significantly better than a power law (likelihood ratio)",
            (),
        )
    # a power law is only *asserted* when it significantly beats BOTH alternatives;
    # "not ruled out" is not enough (CSN), so an indistinguishable fit is inconclusive
    beats_both = (z_ln > 0 and p_ln < 0.1) and (z_exp > 0 and p_exp < 0.1)
    if beats_both:
        return VerificationReport(
            "sound",
            alpha,
            True,
            (Check("power-law-fit", True, detail),),
            None,
            (
                "a power law is plausible and fits better than a lognormal and an "
                "exponential",
            ),
        )
    return VerificationReport(
        "inconclusive",
        alpha,
        True,
        (Check("power-law-fit", True, detail),),
        None,
        (
            "a power law is not ruled out, but cannot be distinguished from a "
            "lognormal or exponential; do not claim 'scale-free' on this evidence",
        ),
    )


def _fit(xs: Sequence[float]) -> tuple[float, float, float, list[float]]:
    """Choose xmin minimising KS distance; return (xmin, alpha, D, tail)."""
    usable = sorted(set(xs))[:-15]  # leave room for a tail of >= 15 points
    # cap the candidate cutoffs to keep the bootstrap (which refits per resample) fast
    if len(usable) > 40:
        step = len(usable) / 40.0
        candidates = [usable[int(i * step)] for i in range(40)]
    else:
        candidates = usable
    best = (xs[0], 2.0, 1.0, list(xs))
    best_d = math.inf
    for xmin in candidates:
        tail = [x for x in xs if x >= xmin]
        if len(tail) < 15:
            continue
        alpha = _alpha(tail, xmin)
        d = _ks(tail, xmin, alpha)
        if d < best_d:
            best_d, best = d, (xmin, alpha, d, tail)
    return best


def _alpha(tail: Sequence[float], xmin: float) -> float:
    """Continuous power-law MLE exponent for a tail above ``xmin``."""
    s = math.fsum(math.log(x / xmin) for x in tail)
    return 1.0 + len(tail) / s if s > 0 else 2.0


def _ks(tail: Sequence[float], xmin: float, alpha: float) -> float:
    """Kolmogorov-Smirnov distance between the tail and the fitted power law."""
    n = len(tail)
    ordered = sorted(tail)
    d = 0.0
    for i, x in enumerate(ordered):
        emp = (i + 1) / n
        model = 1.0 - (x / xmin) ** (1.0 - alpha)  # power-law CDF
        d = max(d, abs(emp - model))
    return d


def _bootstrap_gof(
    xs: Sequence[float], xmin: float, alpha: float, d_obs: float, reps: int
) -> float:
    """Semiparametric bootstrap p-value: share of synthetic KS distances >= observed."""
    body = [x for x in xs if x < xmin]
    n = len(xs)
    ptail = (n - len(body)) / n
    state = 0x9E3779B97F4A7C15 & _MASK
    ge = 0
    for _ in range(reps):
        synth = []
        for _ in range(n):
            state = (_LCG_A * state + _LCG_C) & _MASK
            u = state / _MASK
            if u < ptail or not body:
                state = (_LCG_A * state + _LCG_C) & _MASK
                u2 = (state / _MASK) or 1e-12
                synth.append(xmin * u2 ** (-1.0 / (alpha - 1.0)))
            else:
                state = (_LCG_A * state + _LCG_C) & _MASK
                synth.append(body[state % len(body)])
        _sx, _sa, sd, _stail = _fit(sorted(synth))
        if sd >= d_obs:
            ge += 1
    return ge / reps if reps else 1.0


def _vuong(
    tail: Sequence[float], xmin: float, alpha: float, alt_loglik: Sequence[float]
) -> tuple[float, float]:
    """Normalised log-likelihood ratio (Vuong) of power-law vs alternative; (z, p)."""
    pl = [
        math.log(alpha - 1.0) - math.log(xmin) - alpha * math.log(x / xmin)
        for x in tail
    ]  # tail-normalised power-law log-density
    diffs = [pl[i] - alt_loglik[i] for i in range(len(tail))]
    n = len(diffs)
    mean = math.fsum(diffs) / n
    var = math.fsum((d - mean) ** 2 for d in diffs) / n
    if var <= 0:
        return 0.0, 1.0
    z = math.fsum(diffs) / (math.sqrt(n) * math.sqrt(var))
    return z, math.erfc(abs(z) / math.sqrt(2.0))


def _lognormal_loglik(
    tail: Sequence[float], xmin: float, full: Sequence[float]
) -> list[float]:
    """Per-point log-likelihood of a lognormal *truncated to* the tail above ``xmin``.

    The density must be normalised over ``[xmin, inf)`` so the comparison with the
    (already tail-normalised) power law is fair. The lognormal parameters are estimated
    from the *full* data, not the truncated tail: naive moments of a truncated sample
    are badly biased and would make the lognormal fit poorly, spuriously favouring the
    power law.
    """
    logs = [math.log(x) for x in full]
    n = len(logs)
    mu = math.fsum(logs) / n
    sigma = math.sqrt(math.fsum((v - mu) ** 2 for v in logs) / n) or 1e-9
    norm = math.log(math.sqrt(2 * math.pi) * sigma)
    # log P(X >= xmin) under the fitted lognormal, for the truncation normaliser
    tail_prob = 0.5 * math.erfc((math.log(xmin) - mu) / (sigma * math.sqrt(2.0)))
    log_tail = math.log(tail_prob) if tail_prob > 0 else -700.0
    return [
        -math.log(x) - norm - (math.log(x) - mu) ** 2 / (2 * sigma**2) - log_tail
        for x in tail
    ]


def _exponential_loglik(tail: Sequence[float], xmin: float) -> list[float]:
    """Per-point log-likelihood of an exponential fit to the tail above ``xmin``."""
    mean = math.fsum(tail) / len(tail)
    lam = 1.0 / (mean - xmin) if mean > xmin else 1.0 / mean
    return [math.log(lam) - lam * (x - xmin) for x in tail]
