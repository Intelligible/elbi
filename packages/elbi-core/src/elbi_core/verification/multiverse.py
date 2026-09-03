"""Multiverse / specification-curve engine: a robustness certificate for a claim.

A single dataset admits many defensible analyses, which confounders to control for,
whether to drop outliers, whether to fit on raw values or ranks. A conclusion that holds
under one specification but vanishes under another is not robust; reporting only the
specification that happened to be run is the "garden of forking paths" (Gelman & Loken
2013). This engine implements specification-curve analysis (Simonsohn, Simmons & Nelson
2020) for a claimed effect of ``x`` on ``y``: it enumerates the defensible specification
space, re-estimates the effect under every one, runs the joint resampling-under-the-null
test (is the observed curve more extreme than a no-effect null?), and reports which
analytical decision drives the result via a variance decomposition. Following the
good/bad-controls literature it only varies adjustment for covariates that are
confounders or neutral, never a detected collider, which controlling would spuriously
create an association. The certificate is "the conclusion survives the defensible
multiverse", not "it survived the one analysis we ran".
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ._numerics import _corr, _dot, _frame, _inv, _iqr_mask, _numeric_columns, _role
from ._report import _T

#: At most this many covariates enter the adjustment-set powerset (chosen by relevance
#: to x and y), keeping the multiverse bounded at 2**k adjustment sets.
_MAX_COVARIATES = 3
#: A conclusion is robust when at least this share of specifications support it AND the
#: joint null is rejected; below it (with signal present) the conclusion is fragile.
_ROBUST_SHARE = 0.9
#: The joint test's significance level.
_ALPHA = 0.05
#: Seeded-LCG constants for the deterministic bootstrap (reproducible, no global RNG).
_LCG_A = 6364136223846793005
_LCG_C = 1442695040888963407
_MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class MultiverseReport:
    """A specification-curve robustness certificate for an effect of ``x`` on ``y``."""

    verdict: str  # "robust" | "fragile" | "inconclusive"
    median: float  # median estimate across the specification curve
    dominant_sign: int  # +1 / -1 / 0
    share_supporting: float  # share of specs significant in the dominant direction
    p_value: float  # joint resampling-under-the-null test (Stouffer statistic)
    n_specs: int
    drivers: tuple[tuple[str, float], ...]  # decision -> share of estimate variance
    pivotal: str | None  # the decision the conclusion turns on (when fragile)
    breaking: str | None  # a specification under which the conclusion fails
    curve: tuple[float, ...]  # the sorted estimates (the specification curve)

    def render(self) -> str:
        """Format the robustness certificate as markdown for an agent to read."""
        lines = [
            f"# Robustness certificate: **{self.verdict.upper()}**",
            "",
            f"across {self.n_specs} defensible specifications, the median estimate is "
            f"{self.median:+.3g}; {self.share_supporting:.0%} are significant in the "
            f"same direction (joint test p = {self.p_value:.3g})",
        ]
        if self.curve:
            lines.append(
                f"specification curve: estimate in [{self.curve[0]:+.3g}, "
                f"{self.curve[-1]:+.3g}]"
            )
        if self.drivers:
            top = ", ".join(f"{name} ({share:.0%})" for name, share in self.drivers[:3])
            lines.append(f"what drives the estimate: {top}")
        if self.pivotal:
            lines.append(f"\npivotal decision: {self.pivotal}")
        if self.breaking:
            lines.append(f"breaking specification: {self.breaking}")
        lines.append(
            "\nthe specification set is expert-defensible, not exhaustive; it assumes "
            "the varied covariates are confounders, not mediators"
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class _Spec:
    """One specification: its decisions, source columns, and outlier mask."""

    decisions: dict[str, str]
    controls: tuple[str, ...]
    source: dict[str, list[float]]
    keep: list[bool] | None  # rows to keep (outlier rule), or None to keep all


def verify_multiverse(
    rows: Sequence[dict[str, Any]], x: str, y: str, *, resamples: int = 300
) -> MultiverseReport:
    """Certify how robustly ``x`` affects ``y`` across the analytical multiverse.

    Enumerates specifications over the defensible researcher degrees of freedom, the
    adjustment set (every subset of the relevant confounders, never a detected
    collider), outlier handling (keep or drop IQR-flagged points), and functional form
    (a raw or a rank fit), estimates the effect under each, and runs the joint
    resampling-under-the-null test of Simonsohn et al. (2020) with a seeded bootstrap.
    The verdict is ``robust`` (the joint null is rejected and nearly every specification
    agrees), ``fragile`` (signal is present but a defensible decision overturns it,
    named as the pivotal decision), or ``inconclusive`` (no effect beyond chance across
    the multiverse).
    """
    pool = [c for c in _numeric_columns(rows) if c not in (x, y)]
    data, n = _frame(rows, [x, y, *pool])
    if n < 30 or x not in data or y not in data:
        return MultiverseReport("inconclusive", 0.0, 0, 0.0, 1.0, 0, (), None, None, ())
    pool = [c for c in pool if c in data]
    # only defensible adjustments: confounders or neutral covariates, never colliders
    pool = [z for z in pool if _role(data, x, y, z) != "collider"]
    pool.sort(
        key=lambda z: abs(_corr(data[x], data[z])) + abs(_corr(data[y], data[z])),
        reverse=True,
    )
    covs = pool[:_MAX_COVARIATES]

    ranked = {c: _ranks(data[c]) for c in (x, y, *covs)}
    keep = _iqr_mask(data, (x, y))
    specs = _build_specs(data, ranked, keep, covs)

    idx_all = list(range(n))
    observed = [
        _fit(
            s.source[y],
            s.source[x],
            [s.source[c] for c in s.controls],
            _eff(idx_all, s.keep),
        )
        for s in specs
    ]
    estimates = [b for b, _ in observed]
    significant = [abs(t) > _T for _, t in observed]
    if sum(significant) < max(3, 0.05 * len(specs)):
        return MultiverseReport(
            "inconclusive",
            _median(estimates),
            0,
            0.0,
            1.0,
            len(specs),
            (),
            None,
            None,
            tuple(sorted(estimates)),
        )
    median = _median(estimates)
    dominant = 1 if median > 0 else -1
    share = sum(
        1
        for b, sig in zip(estimates, significant, strict=True)
        if sig and (b > 0) == (dominant > 0)
    ) / len(specs)

    p_value = _joint_null_test(specs, observed, x, y, n, resamples)
    drivers = _variance_decomposition(specs, estimates, covs)
    curve = tuple(sorted(estimates))

    if p_value < _ALPHA and share >= _ROBUST_SHARE:
        verdict, pivotal, breaking = "robust", None, None
    elif p_value < _ALPHA:
        verdict = "fragile"
        pivotal = _pivotal(specs, observed, dominant, covs)
        breaking = _breaking(specs, observed, dominant)
    else:
        verdict, pivotal, breaking = "inconclusive", None, None
    return MultiverseReport(
        verdict,
        median,
        dominant if verdict != "inconclusive" else 0,
        share,
        p_value,
        len(specs),
        drivers,
        pivotal,
        breaking,
        curve,
    )


def _build_specs(
    data: dict[str, list[float]],
    ranked: dict[str, list[float]],
    keep: list[bool],
    covs: Sequence[str],
) -> list[_Spec]:
    """Enumerate the space: adjustment set x outlier handling x functional form."""
    specs: list[_Spec] = []
    for include in _powerset(covs):
        for outliers in ("keep", "drop"):
            for form in ("raw", "rank"):
                src = ranked if form == "rank" else data
                specs.append(
                    _Spec(
                        decisions={
                            **{
                                f"adjust:{z}": ("in" if z in include else "out")
                                for z in covs
                            },
                            "outliers": outliers,
                            "form": form,
                        },
                        controls=tuple(include),
                        source=src,
                        keep=keep if outliers == "drop" else None,
                    )
                )
    return specs


def _eff(indices: Sequence[int], keep: list[bool] | None) -> list[int]:
    """The drawn rows a specification actually uses (its outlier rule applied)."""
    if keep is None:
        return list(indices)
    return [i for i in indices if keep[i]]


def _fit(
    target: Sequence[float],
    xs: Sequence[float],
    ctrls: Sequence[Sequence[float]],
    idx: Sequence[int],
) -> tuple[float, float]:
    """OLS of ``target`` on intercept, ``xs``, controls over rows ``idx``; (coef, t)."""
    if len(idx) <= len(ctrls) + 3:
        return 0.0, 0.0
    design = [[1.0] * len(idx), [xs[i] for i in idx]]
    design += [[c[i] for i in idx] for c in ctrls]
    p, m = len(design), len(idx)
    tv = [target[i] for i in idx]
    inv = _inv([[_dot(design[a], design[b]) for b in range(p)] for a in range(p)])
    if inv is None:
        return 0.0, 0.0
    xty = [_dot(col, tv) for col in design]
    beta = [_dot(inv[a], xty) for a in range(p)]
    resid = [
        tv[k] - math.fsum(beta[a] * design[a][k] for a in range(p)) for k in range(m)
    ]
    s2 = _dot(resid, resid) / max(m - p, 1)
    var = s2 * inv[1][1]
    se = var**0.5 if var > 0 else math.inf
    return beta[1], (beta[1] / se if se > 0 else 0.0)


def _joint_null_test(
    specs: Sequence[_Spec],
    observed: Sequence[tuple[float, float]],
    x: str,
    y: str,
    n: int,
    resamples: int,
) -> float:
    """The resampling-under-the-null joint p-value (Simonsohn et al. 2020).

    For each specification the estimated effect is subtracted to force a true zero
    (``target* = source_y - b_hat * source_x``); the same bootstrap rows are drawn for
    every specification each resample, the whole curve is re-estimated, and the p-value
    is the share of resamples whose Stouffer statistic (mean signed t) is at least as
    extreme as observed.
    """
    obs_stat = abs(math.fsum(t for _, t in observed) / len(observed))
    nulls = [
        [s.source[y][i] - b * s.source[x][i] for i in range(n)]
        for s, (b, _) in zip(specs, observed, strict=True)
    ]
    ge = 0
    for rep in range(resamples):
        draw = _draw(n, rep)
        stat = math.fsum(
            _fit(
                nulls[k],
                s.source[x],
                [s.source[c] for c in s.controls],
                _eff(draw, s.keep),
            )[1]
            for k, s in enumerate(specs)
        ) / len(specs)
        if abs(stat) >= obs_stat - 1e-12:
            ge += 1
    return (1 + ge) / (resamples + 1)


def _variance_decomposition(
    specs: Sequence[_Spec], estimates: Sequence[float], covs: Sequence[str]
) -> tuple[tuple[str, float], ...]:
    """Share of estimate variance each decision explains (eta-squared per factor)."""
    mean = math.fsum(estimates) / len(estimates)
    ss_total = math.fsum((b - mean) ** 2 for b in estimates)
    if ss_total <= 0:
        return ()
    factors = [f"adjust:{z}" for z in covs] + ["outliers", "form"]
    shares = []
    for f in factors:
        groups: dict[str, list[float]] = {}
        for s, b in zip(specs, estimates, strict=True):
            groups.setdefault(s.decisions[f], []).append(b)
        ss_between = math.fsum(
            len(g) * (math.fsum(g) / len(g) - mean) ** 2 for g in groups.values()
        )
        shares.append((_factor_name(f), ss_between / ss_total))
    shares.sort(key=lambda t: t[1], reverse=True)
    return tuple(shares)


def _pivotal(
    specs: Sequence[_Spec],
    observed: Sequence[tuple[float, float]],
    dominant: int,
    covs: Sequence[str],
) -> str | None:
    """Name the decision the conclusion most turns on (largest swing in support)."""

    def rate(factor: str, option: str) -> float:
        hits = [
            abs(t) > _T and (b > 0) == (dominant > 0)
            for s, (b, t) in zip(specs, observed, strict=True)
            if s.decisions[factor] == option
        ]
        return sum(hits) / len(hits) if hits else 0.0

    forks = [(f"adjust:{z}", "in", "out") for z in covs]
    forks += [("outliers", "keep", "drop"), ("form", "raw", "rank")]
    best: tuple[str, str] | None = None
    best_gap = 0.0
    for factor, a, b in forks:
        gap = rate(factor, a) - rate(factor, b)
        if abs(gap) > abs(best_gap):
            best_gap, best = gap, (factor, a if gap > 0 else b)
    if best is None or abs(best_gap) < 0.3:
        return None
    return _describe(best[0], best[1])


def _breaking(
    specs: Sequence[_Spec], observed: Sequence[tuple[float, float]], dominant: int
) -> str | None:
    """Describe a specification under which the conclusion fails (flips or vanishes)."""
    for s, (b, t) in zip(specs, observed, strict=True):
        if not (abs(t) > _T and (b > 0) == (dominant > 0)):
            controls = [
                z
                for z in (
                    k.split(":", 1)[1] for k in s.decisions if k.startswith("adjust:")
                )
                if s.decisions[f"adjust:{z}"] == "in"
            ]
            parts = [f"controlling for {controls}" if controls else "no controls"]
            if s.decisions["outliers"] == "drop":
                parts.append("outliers dropped")
            if s.decisions["form"] == "rank":
                parts.append("rank fit")
            return f"{', '.join(parts)} → estimate {b:+.3g} (not significant)"
    return None


def _factor_name(factor: str) -> str:
    if factor.startswith("adjust:"):
        return f"controlling for '{factor.split(':', 1)[1]}'"
    return {"outliers": "outlier handling", "form": "raw vs rank fit"}.get(
        factor, factor
    )


def _describe(factor: str, supporting: str) -> str:
    """Describe a fork given which option the conclusion needs to hold."""
    if factor.startswith("adjust:"):
        z = factor.split(":", 1)[1]
        return (
            f"the effect only survives when controlling for '{z}'"
            if supporting == "in"
            else f"controlling for '{z}' overturns the effect "
            "(a confounder or mediator)"
        )
    if factor == "outliers":
        return (
            "the effect is driven by outliers (it vanishes once they are dropped)"
            if supporting == "keep"
            else "the effect only appears after dropping outliers"
        )
    return (
        "the effect holds on a raw fit but not a monotone (rank) one"
        if supporting == "raw"
        else "the effect holds only on a rank fit, not a raw one"
    )


def _powerset(items: Sequence[str]) -> list[tuple[str, ...]]:
    """Every subset of ``items`` as a tuple (the adjustment-set multiverse)."""
    out = []
    for bits in range(1 << len(items)):
        out.append(tuple(items[i] for i in range(len(items)) if bits & (1 << i)))
    return out


def _ranks(values: Sequence[float]) -> list[float]:
    """Average ranks of ``values`` (a rank/monotone transform of one column)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _draw(n: int, rep: int) -> list[int]:
    """A deterministic with-replacement bootstrap of ``n`` row indices."""
    state = (rep * 2654435761 + _LCG_C) & _MASK64
    out = []
    for _ in range(n):
        state = (_LCG_A * state + _LCG_C) & _MASK64
        out.append(state % n)
    return out
