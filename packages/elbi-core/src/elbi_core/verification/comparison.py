"""Group-comparison / hypothesis-test gate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ._numerics import (
    _as_float,
    _categorical_columns,
    _cohens_d,
    _iqr_keep,
    _mann_whitney,
    _normal_p,
    _p_two_sided,
    _simpson_reversal,
    _welch,
)
from ._report import (
    _T,
    Check,
    VerificationReport,
)


def verify_comparison(
    rows: Sequence[dict[str, Any]],
    group: str,
    value: str,
    *,
    subgroup: str | None = None,
    family_size: int = 1,
) -> VerificationReport:
    """Verify that two groups differ on ``value``, soundly.

    Splits ``rows`` by the two levels of ``group`` and tests the mean difference in
    ``value`` (Welch). It checks the difference is non-trivial in size, survives outlier
    removal, holds under the assumption-appropriate test when the samples are small and
    non-normal (Mann-Whitney, so a parametric artifact is caught), survives Bonferroni
    correction when it is one of ``family_size`` comparisons, and, given a ``subgroup``
    column, does not reverse within subgroups (Simpson's paradox). The verdict is
    ``sound`` (every check held), ``unsound`` (one broke it), or ``inconclusive`` (no
    significant difference).
    """
    triples: list[tuple[str, float, str]] = []
    for row in rows:
        label = str(row.get(group, "")).strip()
        v = _as_float(row.get(value))
        if not label or v is None:
            continue
        sub = str(row.get(subgroup, "")).strip() if subgroup else ""
        triples.append((label, v, sub))
    labels = sorted({g for g, _, _ in triples})
    if len(labels) != 2:
        return VerificationReport(
            "inconclusive",
            0.0,
            False,
            (),
            None,
            (f"a comparison needs two groups in '{group}', found {len(labels)}",),
        )
    a, b = labels
    av = [v for g, v, _ in triples if g == a]
    bv = [v for g, v, _ in triples if g == b]
    if len(av) < 10 or len(bv) < 10:
        return VerificationReport(
            "inconclusive", 0.0, False, (), None, ("too few rows in a group",)
        )
    diff, t = _welch(av, bv)
    if abs(t) <= _T:
        return VerificationReport(
            "inconclusive",
            diff,
            False,
            (),
            None,
            (f"no significant difference in '{value}' between '{a}' and '{b}'",),
        )
    positive = diff > 0
    checks: list[Check] = []
    pivotal: str | None = None

    d = _cohens_d(av, bv)
    # Cohen's d badly understates a binary/proportion metric (a large relative
    # difference in rates has a small d), so for a 0/1 value the size is reported but
    # not used to fail the result; for a continuous value a negligible d still flags.
    binary = set(av) | set(bv) <= {0.0, 1.0}
    sizable = binary or abs(d) >= 0.2
    checks.append(
        Check(
            "effect-size",
            sizable,
            f"Cohen's d = {d:+.2f}" + ("" if sizable else " (negligible)"),
        )
    )
    if not sizable:
        pivotal = "the difference is significant but negligible in size"

    # assumption-appropriate test: a parametric difference on a small, non-normal
    # sample must also hold under the rank-based Mann-Whitney test, or it is an
    # artifact of the normality assumption rather than a real difference.
    if min(len(av), len(bv)) < 30:
        pa, pb = _normal_p(av), _normal_p(bv)
        non_normal = (pa is not None and pa < 0.05) or (pb is not None and pb < 0.05)
        if non_normal:
            _, zu = _mann_whitney(av, bv)
            held = abs(zu) > _T
            checks.append(
                Check(
                    "distribution",
                    held,
                    "difference holds under Mann-Whitney (data is non-normal)"
                    if held
                    else "non-normal data: the difference fails the rank-based test",
                )
            )
            if not held:
                pivotal = pivotal or (
                    "the groups are non-normal and the difference does not survive "
                    "the assumption-free Mann-Whitney test"
                )

    # multiple comparisons: one significant result out of many is expected by
    # chance, so a declared family of tests must survive Bonferroni correction.
    if family_size > 1:
        p = _p_two_sided(t)
        survives = p < 0.05 / family_size
        checks.append(
            Check(
                "multiplicity",
                survives,
                f"survives Bonferroni for {family_size} tests (p = {p:.2g})"
                if survives
                else f"does not survive correction for {family_size} comparisons",
            )
        )
        if not survives:
            pivotal = pivotal or (
                f"the difference is not significant after correcting for "
                f"{family_size} comparisons"
            )

    aa = [v for v, k in zip(av, _iqr_keep(av), strict=True) if k]
    bb = [v for v, k in zip(bv, _iqr_keep(bv), strict=True) if k]
    if len(aa) >= 10 and len(bb) >= 10:
        d2, t2 = _welch(aa, bb)
        held = abs(t2) > _T and (d2 > 0) == positive
        checks.append(
            Check(
                "outliers",
                held,
                "survives IQR outlier removal"
                if held
                else "difference vanishes without outliers",
            )
        )
        if not held:
            pivotal = pivotal or "the difference is driven by outliers"

    # Simpson's paradox: scan the named subgroup *and* every categorical column for a
    # within-group reversal, so the check does not depend on the caller naming the right
    # column, the overall difference is unsound if it flips inside any of them.
    candidates = [subgroup] if subgroup else []
    candidates += _categorical_columns(rows, exclude=(group, value, subgroup))
    if candidates:
        reversal = _simpson_reversal(rows, group, value, a, b, positive, candidates)
        if reversal is None:
            checks.append(Check("subgroup-consistency", True, "holds within subgroups"))
        else:
            col, level = reversal
            checks.append(
                Check(
                    "subgroup-consistency",
                    False,
                    f"reverses within '{col}' = {level}",
                )
            )
            # a sign reversal is the most serious flaw here, so it leads the report
            # even when a lesser check (e.g. a small effect size) also fired
            pivotal = (
                f"Simpson's paradox: the difference reverses within '{col}' = "
                f"{level}; the overall comparison is confounded by it"
            )

    verdict = "sound" if all(c.survived for c in checks) else "unsound"
    caveats = (
        f"compares '{a}' and '{b}' as observed; a difference is not by itself a cause",
    )
    return VerificationReport(verdict, diff, True, tuple(checks), pivotal, caveats)
