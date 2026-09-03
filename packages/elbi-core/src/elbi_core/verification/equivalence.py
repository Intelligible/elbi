"""Equivalence-testing gate (TOST): soundly concluding "no meaningful difference"."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._numerics import _as_float, _mean_var, _t_sf
from ._report import Check, VerificationReport


def verify_equivalence(
    rows: Sequence[dict[str, Any]],
    group: str,
    value: str,
    *,
    sesoi: float,
) -> VerificationReport:
    """Verify two groups are *equivalent* on ``value`` within ±``sesoi``, soundly.

    A non-significant difference is not evidence of equivalence; that is the
    absence-of-evidence fallacy. The honest test is two one-sided tests (TOST,
    Schuirmann 1987): equivalence is concluded only when the difference is significantly
    *inside* the smallest effect of interest ``sesoi`` from both sides. Because the
    bound is a domain judgement the data cannot supply, ``sesoi`` must be given (in the
    units of ``value``). The verdict is ``sound`` (statistically equivalent),
    ``unsound`` (a real difference *larger* than ``sesoi``), or ``inconclusive``
    (neither: typically underpowered).
    """
    if sesoi <= 0:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            (
                "equivalence needs a positive smallest-effect-of-interest (sesoi) "
                "bound; it cannot be inferred from the data",
            ),
        )
    buckets: dict[str, list[float]] = {}
    for row in rows:
        label = str(row.get(group, "")).strip()
        v = _as_float(row.get(value))
        if label and v is not None:
            buckets.setdefault(label, []).append(v)
    labels = sorted(g for g, vs in buckets.items() if len(vs) >= 10)
    if len(labels) != 2:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("equivalence needs exactly two groups of at least 10",),
        )
    a, b = labels
    av, bv = buckets[a], buckets[b]
    ma, va = _mean_var(av)
    mb, vb = _mean_var(bv)
    diff = ma - mb
    se = math.sqrt(va / len(av) + vb / len(bv))
    if se <= 0:
        return VerificationReport(
            "inconclusive", diff, False, (), None, ("zero variance; cannot test",)
        )
    df = (va / len(av) + vb / len(bv)) ** 2 / (
        (va / len(av)) ** 2 / (len(av) - 1) + (vb / len(bv)) ** 2 / (len(bv) - 1)
    )
    # two one-sided tests against the equivalence bounds [-sesoi, +sesoi]
    p_lower = _one_sided(((diff + sesoi) / se), df, greater=True)
    p_upper = _one_sided(((diff - sesoi) / se), df, greater=False)
    equivalent = max(p_lower, p_upper) < 0.05
    # a conventional difference test, to separate "a real, larger difference" from
    # "merely underpowered" when equivalence is not concluded
    differs = _t_sf(diff / se, df) < 0.05 and abs(diff) > sesoi

    checks = (
        Check(
            "equivalence",
            equivalent,
            f"both one-sided tests reject (p = {max(p_lower, p_upper):.2g})"
            if equivalent
            else f"not bounded within ±{sesoi:g} (p = {max(p_lower, p_upper):.2g})",
        ),
    )
    if equivalent:
        verdict = "sound"
    elif differs:
        verdict = "unsound"
    else:
        verdict = "inconclusive"
    caveats = (
        f"difference between '{a}' and '{b}' is {diff:+.3g} (within a ±{sesoi:g} "
        "bound); equivalence is relative to that bound, a domain judgement",
    )
    return VerificationReport(verdict, diff, equivalent, checks, None, caveats)


def _one_sided(t: float, df: float, *, greater: bool) -> float:
    """One-sided Student-t p-value; ``greater`` tests the upper tail, else lower."""
    half = _t_sf(t, df) / 2.0
    upper = half if t > 0 else 1.0 - half
    return upper if greater else 1.0 - upper
