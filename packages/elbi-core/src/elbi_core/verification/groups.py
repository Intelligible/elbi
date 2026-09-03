"""k-group comparison gate (ANOVA across three or more groups)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._numerics import _as_float, _chi2_sf, _f_sf, _mean_var
from ._report import Check, VerificationReport


def verify_groups(
    rows: Sequence[dict[str, Any]], group: str, value: str
) -> VerificationReport:
    """Verify that three or more groups differ on ``value``, soundly.

    Splits ``rows`` by the levels of ``group`` and tests for any difference in ``value``
    with Welch's ANOVA (which, unlike classic ANOVA, does not assume equal variances:
    the modern default). A parametric difference must also hold under the rank-based
    Kruskal-Wallis test, or it is an artifact of the normality/variance assumptions
    rather than a real difference. The verdict is ``sound`` (a difference that survives
    the assumption-free test, with the differing pairs named), ``unsound``
    (parametric-only, so an assumption artifact), or ``inconclusive`` (no difference, or
    fewer than three groups: use ``verify_comparison`` for two).
    """
    buckets: dict[str, list[float]] = {}
    for row in rows:
        label = str(row.get(group, "")).strip()
        v = _as_float(row.get(value))
        if label and v is not None:
            buckets.setdefault(label, []).append(v)
    groups = {g: vs for g, vs in buckets.items() if len(vs) >= 5}
    if len(groups) < 3:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            (
                "a k-group test needs three or more groups of at least 5 (use "
                "verify_comparison for two)",
            ),
        )

    f_stat, df1, df2 = _welch_anova(groups)
    p = _f_sf(f_stat, df1, df2)
    if p >= 0.05:
        return VerificationReport(
            "inconclusive",
            f_stat,
            False,
            (),
            None,
            (f"no significant difference in '{value}' across {len(groups)} groups",),
        )

    eta2 = _eta_squared(groups)
    checks: list[Check] = []
    pivotal: str | None = None

    h_stat, df_k = _kruskal(groups)
    held = _chi2_sf(h_stat, df_k) < 0.05
    checks.append(
        Check(
            "distribution",
            held,
            "difference holds under Kruskal-Wallis"
            if held
            else "the difference fails the rank-based Kruskal-Wallis test",
        )
    )
    if not held:
        pivotal = (
            "the groups differ under the parametric test but not the assumption-free "
            "Kruskal-Wallis test, so the difference is an artifact of its assumptions"
        )

    differing = _dunn_pairs(groups)
    verdict = "sound" if all(c.survived for c in checks) else "unsound"
    pairs = ", ".join(f"{a}≠{b}" for a, b in differing) or "none at Bonferroni"
    caveats = (
        f"omnibus across {len(groups)} groups (eta-squared = {eta2:.2f}); differing "
        f"pairs (Dunn, Bonferroni): {pairs}",
    )
    return VerificationReport(verdict, f_stat, True, tuple(checks), pivotal, caveats)


def _welch_anova(groups: dict[str, list[float]]) -> tuple[float, float, float]:
    """Welch's ANOVA F statistic and its two (non-integer) degrees of freedom."""
    k = len(groups)
    n = {g: len(v) for g, v in groups.items()}
    stats = {g: _mean_var(v) for g, v in groups.items()}
    w = {g: (n[g] / var if var > 0 else 0.0) for g, (_, var) in stats.items()}
    w_sum = math.fsum(w.values())
    if w_sum <= 0:
        return 0.0, k - 1, 1.0
    grand = math.fsum(w[g] * stats[g][0] for g in groups) / w_sum
    num = math.fsum(w[g] * (stats[g][0] - grand) ** 2 for g in groups) / (k - 1)
    s = math.fsum(
        (1.0 / (n[g] - 1)) * (1.0 - w[g] / w_sum) ** 2 for g in groups if n[g] > 1
    )
    denom = 1.0 + (2.0 * (k - 2) / (k * k - 1)) * s
    f_stat = num / denom if denom > 0 else 0.0
    df2 = (k * k - 1) / (3.0 * s) if s > 0 else math.inf
    return f_stat, float(k - 1), df2


def _eta_squared(groups: dict[str, list[float]]) -> float:
    """Proportion of variance in the value explained by group membership."""
    allv = [v for vs in groups.values() for v in vs]
    grand = math.fsum(allv) / len(allv)
    ss_between = math.fsum(
        len(vs) * (math.fsum(vs) / len(vs) - grand) ** 2 for vs in groups.values()
    )
    ss_total = math.fsum((v - grand) ** 2 for v in allv)
    return ss_between / ss_total if ss_total > 0 else 0.0


def _pooled_ranks(
    groups: dict[str, list[float]],
) -> tuple[dict[str, list[float]], list[int]]:
    """Average ranks of every value, regrouped, plus the tie-group sizes."""
    pooled = sorted((v, g) for g, vs in groups.items() for v in vs)
    ranks = [0.0] * len(pooled)
    ties: list[int] = []
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg
        if j > i:
            ties.append(j - i + 1)
        i = j + 1
    by_group: dict[str, list[float]] = {g: [] for g in groups}
    for (_, g), r in zip(pooled, ranks, strict=True):
        by_group[g].append(r)
    return by_group, ties


def _kruskal(groups: dict[str, list[float]]) -> tuple[float, int]:
    """Kruskal-Wallis H (tie-corrected) and its degrees of freedom."""
    ranked, ties = _pooled_ranks(groups)
    big_n = sum(len(v) for v in groups.values())
    h = 12.0 / (big_n * (big_n + 1)) * math.fsum(
        math.fsum(rs) ** 2 / len(rs) for rs in ranked.values()
    ) - 3.0 * (big_n + 1)
    correction = 1.0 - math.fsum(t**3 - t for t in ties) / (big_n**3 - big_n)
    return (h / correction if correction > 0 else h), len(groups) - 1


def _dunn_pairs(groups: dict[str, list[float]]) -> list[tuple[str, str]]:
    """Pairs that differ by Dunn's post-hoc test with Bonferroni correction."""
    ranked, ties = _pooled_ranks(groups)
    big_n = sum(len(v) for v in groups.values())
    mean_rank = {g: math.fsum(rs) / len(rs) for g, rs in ranked.items()}
    tie_adj = math.fsum(t**3 - t for t in ties) / (12.0 * (big_n - 1))
    labels = sorted(groups)
    m = len(labels) * (len(labels) - 1) // 2
    out: list[tuple[str, str]] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            na, nb = len(groups[a]), len(groups[b])
            se = math.sqrt(
                (big_n * (big_n + 1) / 12.0 - tie_adj) * (1.0 / na + 1.0 / nb)
            )
            if se <= 0:
                continue
            z = abs(mean_rank[a] - mean_rank[b]) / se
            if math.erfc(z / math.sqrt(2.0)) < 0.05 / m:
                out.append((a, b))
    return out
