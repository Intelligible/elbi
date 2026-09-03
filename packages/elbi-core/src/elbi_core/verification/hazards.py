"""Proportional-hazards gate: is a Cox hazard ratio meaningful here?

A Cox model summarises a group difference in survival as a single hazard ratio, but that
number only means something if the hazards are *proportional*; the ratio is constant
over time. When survival curves cross or converge, no single hazard ratio describes
them, and reporting one is misleading. Fitting the Cox partial likelihood to test this
is heavy; the stdlib-feasible screen is to build the Kaplan-Meier survival curves for
the two groups and detect whether they cross, the clearest signature of a time-varying
(non-proportional) effect.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._numerics import _as_float
from ._report import Check, VerificationReport


def verify_proportional_hazards(
    rows: Sequence[dict[str, Any]], time: str, event: str, group: str
) -> VerificationReport:
    """Verify hazards are proportional between two groups (a Cox HR is meaningful).

    Builds the Kaplan-Meier survival curves for the two groups and checks whether they
    cross. The verdict is ``unsound`` (the curves cross: hazards are non-proportional,
    so a single hazard ratio misleads; report time-specific survival instead), ``sound``
    (the curves do not cross, hazards look proportional), or ``inconclusive`` (too few
    events or not two groups).
    """
    by_group: dict[str, list[tuple[float, int]]] = {}
    for row in rows:
        t = _as_float(row.get(time))
        e = _as_float(row.get(event))
        g = str(row.get(group, "")).strip()
        if t is not None and e is not None and g:
            by_group.setdefault(g, []).append((t, round(e)))
    groups = sorted(g for g, v in by_group.items() if len(v) >= 20)
    if len(groups) != 2:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            ("proportional-hazards needs exactly two groups of at least 20",),
        )
    a, b = groups
    crosses = _curves_cross(by_group[a], by_group[b])
    checks = (
        Check(
            "proportional-hazards",
            not crosses,
            "survival curves cross (non-proportional)"
            if crosses
            else "survival curves do not cross",
        ),
    )
    pivotal = (
        (
            f"the survival curves for '{a}' and '{b}' cross, so the hazards are not "
            "proportional: a single Cox hazard ratio misleads; report survival at "
            "specific times instead"
        )
        if crosses
        else None
    )
    verdict = "unsound" if crosses else "sound"
    return VerificationReport(verdict, 0.0, True, checks, pivotal, ())


def _km(data: Sequence[tuple[float, int]]) -> list[tuple[float, float]]:
    """Kaplan-Meier survival curve as (time, survival) steps."""
    times = sorted({t for t, _ in data})
    surv = 1.0
    out: list[tuple[float, float]] = []
    for t in times:
        at_risk = sum(1 for ti, _ in data if ti >= t)
        events = sum(1 for ti, e in data if ti == t and e == 1)
        if at_risk > 0 and events > 0:
            surv *= 1.0 - events / at_risk
        out.append((t, surv))
    return out


def _survival_at(curve: Sequence[tuple[float, float]], t: float) -> float:
    """Survival at time ``t`` from a step curve (last step at or before ``t``)."""
    s = 1.0
    for ti, si in curve:
        if ti <= t:
            s = si
        else:
            break
    return s


def _curves_cross(
    da: Sequence[tuple[float, int]], db: Sequence[tuple[float, int]]
) -> bool:
    """Whether the two Kaplan-Meier curves cross by a non-trivial margin."""
    ca, cb = _km(da), _km(db)
    grid = sorted({t for t, _ in ca} | {t for t, _ in cb})
    if not grid:
        return False
    upper = grid[int(0.9 * (len(grid) - 1))]  # ignore the noisy far tail
    signs = []
    for t in grid:
        if t > upper:
            break
        diff = _survival_at(ca, t) - _survival_at(cb, t)
        if abs(diff) > 0.05:  # only count a clear separation
            signs.append(1 if diff > 0 else -1)
    return len(set(signs)) > 1
